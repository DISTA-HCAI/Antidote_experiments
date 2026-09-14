"""
Il ciclo di immunizzazione bi-livello (AntiDote).

Ogni BLOCCO alterna due fasi:

  Fase 1 - ADVERSARY: l'hypernetwork guarda le attivazioni interne del defender su un
           prompt dannoso e impara a sintetizzare una patch LoRA che, una volta iniettata,
           spinge il modello a preferire la risposta DANNOSA ('chosen' nel dataset grezzo)
           rispetto a quella sicura -- cioe' simula un fine-tuner malevolo.

  Fase 2 - DEFENDER: l'adapter LoRA del defender viene aggiornato in modo che, ANCHE con
           quella stessa patch iniettata, continui a preferire la risposta sicura; il
           tutto restando vicino al comportamento originale sulle istruzioni benigne
           (loss LM + loss KL contro il modello di riferimento congelato).

Correzioni rispetto a `train_tamper_resistant_model` di training.py (gia' applicate nel
notebook e mantenute qui):

  1. Gradienti morti / che colavano -- l'originale non congelava i parametri del defender
     durante la fase dell'adversary (ne' viceversa) e non chiamava `optimizer_d.zero_grad()`
     in quella fase, quindi gradienti vecchi del defender venivano applicati di soppiatto
     al primo passo del blocco successivo. Qui si commuta esplicitamente `requires_grad`
     per fase e si azzerano i gradienti ad ogni passo.
  2. Hook che si accumulavano -- `with act_cache:` veniva rientrato ad ogni passo e
     `__exit__` non faceva nulla, quindi i forward hook crescevano senza limite. Qui gli
     hook si registrano UNA volta e ad ogni passo si azzerano solo i valori.
  3. Preprocessing mancante / chiavi sbagliate -- la fase del defender leggeva campi
     inesistenti ('safe_input_ids'/'harmful_input_ids') e passava le stringhe grezze del
     batch benigno come kwargs del modello. Qui entrambi i batch passano da
     `preprocess_for_dpo` / `preprocess_for_it`.
  4. Orientamento chosen/rejected -- nel dataset 'chosen' e' la risposta DANNOSA e
     'rejected' quella SICURA. La loss di sicurezza del defender vuole l'orientamento
     opposto: qui lo si costruisce esplicitamente scambiando le due meta'.
  5. bf16 / gradient checkpointing / `Accelerator` multi-GPU -- rimossi: questo e' un
     singolo processo, il device arriva dal config.

Correzioni ULTERIORI rispetto al notebook (che aveva ereditato questi due bug da
training.py e non li aveva visti):

  6. L'adversary girava sotto `torch.no_grad()` -> gradiente identicamente 0, mai
     allenato. Vedi `make_adversarial_patch`.
  7. `model.set_adapter()` ad ogni patch riaccendeva `requires_grad` sul defender ->
     il backward della "fase adversary" finiva sul defender. Vedi il commento prima del
     ciclo.
  8. Gradienti inf/NaN (con questa scala di loss capita) corrompevano i pesi tramite
     `clip_grad_norm_`: ora il passo viene saltato e contato (`skipped` nel JSONL).

Scelte per contenere la RAM (il notebook teneva TRE copie del modello in memoria):

  - il modello di riferimento e' di default "shared": stesso PeftModel con l'adapter
    spento (vedi model_setup.ReferenceModel), zero copie in piu';
  - le log-prob di riferimento si calcolano PRIMA di iniettare la patch adversariale;
  - i tre termini della loss del defender fanno backward separatamente, cosi' in RAM
    vive il grafo di un solo forward alla volta;
  - il monitor controlla la RAM disponibile ad ogni passo e interrompe la run (salvando
    lo stato) prima che sia il sistema operativo a uccidere qualcosa a caso.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .activations import ActivationCache
from .adversary import Adversary
from .config import Config
from .data import preprocess_for_dpo, preprocess_for_it, to_device
from .injection import adversarial_patch
from .losses import DPOLoss, compute_dpo_loss, compute_kl_loss, compute_lm_loss, compute_reference_logps
from .monitor import RunMonitor


class TimeBudgetExceeded(Exception):
    """Sollevata (e gestita) dentro `run_immunisation` quando scade `training.max_minutes`."""


class ProbeAbort(Exception):
    """Sollevata da `on_block_end` quando la sonda periodica dice che il trial e' perso.

    Non viene gestita qui: risale fino a `run.py`, che chiude la run con status
    `aborted_probe` salvando comunque le metriche gia' raccolte.
    """


def _cycle(dataloader: DataLoader) -> Iterator[Dict[str, List[str]]]:
    """
    Itera all'infinito su un DataLoader: quando finisce, ricomincia da capo.

    Nell'originale questo era gestito a mano con try/except StopIteration, con un bug
    sottile: nella fase del defender i due iteratori (dannoso e benigno) venivano
    ricreati ENTRAMBI quando si esauriva il primo, perdendo un batch dell'altro. Con due
    generatori indipendenti il problema non si pone.
    """
    while True:
        for batch in dataloader:
            yield batch


def total_planned_steps(cfg: Config) -> int:
    """Quanti passi in totale fara' la run: serve alla barra di avanzamento e all'ETA.
    Ogni blocco fa k_steps passi di adversary + k_steps passi di defender."""
    t = cfg.training
    return int(t["epochs"]) * int(t["num_blocks"]) * int(t["k_steps"]) * 2


def set_requires_grad(params, flag: bool) -> None:
    for p in params:
        p.requires_grad = flag


def run_immunisation(
    cfg: Config,
    model,
    adversary: Adversary,
    ref_model,
    tokenizer,
    defender_params: List[torch.nn.Parameter],
    harmful_dataloader: DataLoader,
    benign_dataloader: DataLoader,
    monitor: RunMonitor,
    on_block_end: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Esegue il ciclo bi-livello completo. Ritorna lo storico delle loss.

    Il modello viene modificato sul posto: alla fine, `model` e' il defender immunizzato
    (PeftModel NON fuso, cosi' si puo' salvare il solo adapter) e `adversary` e'
    l'hypernetwork allenato.
    """
    device = cfg.device
    t = cfg.training
    d = cfg.data

    k_steps = int(t["k_steps"])
    num_blocks = int(t["num_blocks"])
    epochs = int(t["epochs"])
    max_length = int(d["max_length"])
    label_pad = int(d.get("label_pad_token_id", -100))
    grad_clip = float(t.get("grad_clip", 1.0))
    skip_nonfinite = bool(t.get("skip_step_on_nonfinite_grad", True))
    skipped_steps = {"adversary": 0, "defender": 0}
    weights = t["defender_loss_weights"]
    adapter_name = cfg.model.get("adapter_name", "defender")

    model.to(device)
    adversary.to(device)

    optimizer_d = AdamW(defender_params, lr=float(t["lr_defender"]))
    optimizer_a = AdamW(adversary.parameters(), lr=float(t["lr_adversary"]))

    # Una sola istanza di DPOLoss, costruita con beta/loss_type dal config (nell'originale
    # erano hard-coded dentro compute_dpo_loss).
    dpo_cfg = t["dpo"]
    dpo_util = DPOLoss(
        beta=float(dpo_cfg["beta"]),
        label_smoothing=float(dpo_cfg.get("label_smoothing", 0.0)),
        loss_type=str(dpo_cfg["loss_type"]),
        device=device,
        average_log_prob=bool(dpo_cfg.get("average_log_prob", False)),
    )

    # Budget di tempo (minuti) per il solo ciclo di training: 0/null = illimitato. Quando
    # scade, il ciclo si ferma in modo pulito a fine passo e la run prosegue con
    # checkpoint e valutazione: serve a rendere prevedibile il costo di un trial HPO.
    max_minutes = float(t.get("max_minutes", 0) or 0)
    t_start = time.time()

    def time_budget_exhausted() -> bool:
        return bool(max_minutes) and (time.time() - t_start) / 60.0 >= max_minutes

    target_modules = list(t["target_modules_to_attack"])

    def make_adversarial_patch(input_ids, attention_mask, act_cache: ActivationCache, train_adversary: bool):
        """
        Genera la patch LoRA malevola per il batch corrente: un forward pass "di
        osservazione" per catturare le attivazioni, poi l'adversary le trasforma in
        coppie (U, V) layer per layer.

        BUG DELL'ORIGINALE (notebook e training.py), corretto qui: TUTTO il corpo di
        questa funzione stava sotto `torch.no_grad()`, adversary compreso. Le matrici
        (U, V) uscivano quindi senza grafo, `loss_a.backward()` non raggiungeva mai
        l'adversary (gradiente = 0, verificato) e `optimizer_a.step()` era un no-op:
        l'adversary restava all'inizializzazione casuale per tutta la run. Qui il
        forward di osservazione resta senza grad (giusto: serve solo a leggere le
        attivazioni), ma l'adversary gira con il grad attivo nella sua fase
        (`train_adversary=True`) e senza nella fase del defender (dove e' congelato:
        risparmia anche memoria, perche' le (U, V) non portano grafo).
        """
        act_cache.clear_cache()

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        lora_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        with torch.enable_grad() if train_adversary else torch.no_grad():
            for layer_name, activations in act_cache.activations.items():
                if activations is None:
                    continue
                # L'ultimo pezzo del nome (es. "q_proj") sceglie le teste dell'adversary.
                config_name = layer_name.split(".")[-1]
                U_gen, V_gen = adversary(activations.to(device), config_name)
                lora_weights[layer_name] = (U_gen, V_gen)
        return lora_weights

    def clip_or_skip(params, optimizer) -> Tuple[float, bool]:
        """
        Clippa la norma del gradiente e fa il passo dell'optimizer; se la norma non e'
        finita (inf/NaN) salta il passo e azzera i gradienti, ritornando skipped=True.
        Senza questa guardia `clip_grad_norm_` moltiplicherebbe i gradienti per
        max_norm/inf = 0, e inf*0 = NaN finirebbe dritto nei pesi.
        """
        grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        if skip_nonfinite and not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            return float(grad_norm), True
        optimizer.step()
        return float(grad_norm), False

    # BUG DELL'ORIGINALE, corretto qui: `model.set_adapter(...)` veniva richiamato ad
    # OGNI generazione di patch, e in PEFT `set_adapter` rimette `requires_grad=True` su
    # tutti i tensori dell'adapter. Risultato: nella fase dell'adversary il backward
    # finiva sul DEFENDER (gradienti ~1e18, poi scartati dallo zero_grad successivo).
    # Qui l'adapter si attiva una volta sola e `requires_grad` e' governato SOLO dalle
    # due fasi qui sotto.
    model.set_adapter(adapter_name)

    harmful_iterator = _cycle(harmful_dataloader)
    benign_iterator = _cycle(benign_dataloader)

    # Gli hook si registrano UNA volta sola, e il context manager li rimuove comunque
    # all'uscita (anche in caso di eccezione o Ctrl-C).
    stopped_by_time_budget = False
    try:
        with ActivationCache(model, target_modules) as act_cache:
            n_hooks = len(act_cache.hooks)
            monitor.log_line(f"[setup] hook di attivazione registrati su {n_hooks} moduli {target_modules}")

            for epoch in range(1, epochs + 1):
                monitor.log_line(f"\n===== Epoca {epoch}/{epochs} =====", style="bold cyan")

                for block_idx in range(1, num_blocks + 1):
                    # ============ FASE 1: ALLENAMENTO DELL'ADVERSARY ============
                    monitor.log_line(
                        f"--- Blocco {block_idx}/{num_blocks}: adversary per {k_steps} passo/i ---",
                        style="red",
                    )
                    adversary.train()
                    # Il defender non si tocca in questa fase: congelato. Il notebook lo
                    # metteva in eval() per avere un bersaglio deterministico (niente dropout
                    # LoRA); pero' transformers applica il gradient checkpointing SOLO in
                    # modalita' train(). Quindi: train() per il checkpointing, e i soli moduli
                    # Dropout rimessi in eval() -> stesso risultato numerico di eval(), meno RAM.
                    model.train()
                    for module in model.modules():
                        if isinstance(module, torch.nn.Dropout):
                            module.eval()
                    set_requires_grad(model.parameters(), False)
                    set_requires_grad(adversary.parameters(), True)

                    for step in range(1, k_steps + 1):
                        harmful_batch = next(harmful_iterator)

                        # NOTA: in questo dataset 'chosen' = risposta dannosa, 'rejected' =
                        # risposta sicura. L'orientamento grezzo qui sotto e' quindi
                        # esattamente quello che vuole l'adversary: preferire 'chosen'.
                        adv_batch = to_device(
                            preprocess_for_dpo(harmful_batch, tokenizer, max_length, label_pad), device
                        )

                        # Log-prob di riferimento PRIMA di iniettare la patch: in modalita'
                        # "shared" il riferimento e' lo stesso modello con l'adapter spento,
                        # e non deve vedere la perturbazione adversariale.
                        monitor.check_memory()
                        ref_logps = compute_reference_logps(ref_model, adv_batch, dpo_util)

                        lora_weights_adv = make_adversarial_patch(
                            adv_batch["chosen_input_ids"], adv_batch["chosen_attention_mask"],
                            act_cache, train_adversary=True,
                        )

                        # Azzeriamo entrambi gli optimizer PRIMA del backward: cosi' nessun
                        # gradiente vecchio del defender sopravvive fino al primo passo della
                        # Fase 2 (bug #1 dell'originale).
                        optimizer_a.zero_grad(set_to_none=True)
                        optimizer_d.zero_grad(set_to_none=True)

                        # La patch resta iniettata solo per il calcolo della loss; il
                        # ripristino e' garantito anche se qualcosa esplode dentro al blocco.
                        # Il backward avviene qui dentro cosi' il grafo viene liberato subito.
                        monitor.check_memory()
                        with adversarial_patch(model, lora_weights_adv):
                            loss_a = compute_dpo_loss(model, adv_batch, ref_logps, dpo_util)
                            loss_a.backward()
                        # Le (U, V) portano il grafo dell'adversary: liberiamolo subito.
                        del lora_weights_adv

                        grad_norm, skipped = clip_or_skip(adversary.parameters(), optimizer_a)
                        if skipped:
                            skipped_steps["adversary"] += 1
                            monitor.log_line(
                                f"[adversary] e{epoch} b{block_idx} s{step}: gradiente non finito "
                                f"({grad_norm}), passo saltato (#{skipped_steps['adversary']})",
                                style="yellow",
                            )

                        monitor.record(
                            phase="adversary",
                            epoch=epoch,
                            block=block_idx,
                            step=step,
                            k_steps=k_steps,
                            metrics={"adversary_loss": loss_a.item()},
                            extra={"grad_norm": grad_norm, "skipped": int(skipped)},
                        )
                        if time_budget_exhausted():
                            raise TimeBudgetExceeded(max_minutes)

                    # ============ FASE 2: ALLENAMENTO DEL DEFENDER ============
                    monitor.log_line(
                        f"--- Blocco {block_idx}/{num_blocks}: defender per {k_steps} passo/i ---",
                        style="green",
                    )
                    adversary.eval()
                    model.train()
                    set_requires_grad(adversary.parameters(), False)
                    # Solo l'adapter del defender e' allenabile: il modello base resta congelato.
                    for n, p in model.named_parameters():
                        p.requires_grad = adapter_name in n

                    for step in range(1, k_steps + 1):
                        harmful_batch = next(harmful_iterator)
                        benign_batch_raw = next(benign_iterator)

                        dpo_batch = to_device(
                            preprocess_for_dpo(harmful_batch, tokenizer, max_length, label_pad), device
                        )
                        benign_batch = to_device(
                            preprocess_for_it(benign_batch_raw, tokenizer, max_length, label_pad), device
                        )

                        # Obiettivo di sicurezza: sotto la patch adversariale, il defender
                        # deve COMUNQUE preferire la risposta sicura -- cioe' chosen/rejected
                        # vanno scambiati rispetto all'orientamento grezzo del dataset.
                        defender_dpo_batch = {
                            "chosen_input_ids": dpo_batch["rejected_input_ids"],
                            "chosen_attention_mask": dpo_batch["rejected_attention_mask"],
                            "chosen_labels": dpo_batch["rejected_labels"],
                            "rejected_input_ids": dpo_batch["chosen_input_ids"],
                            "rejected_attention_mask": dpo_batch["chosen_attention_mask"],
                            "rejected_labels": dpo_batch["chosen_labels"],
                        }
                        # Riferimento PRIMA della patch (vedi Fase 1).
                        monitor.check_memory()
                        ref_logps = compute_reference_logps(ref_model, defender_dpo_batch, dpo_util)

                        lora_weights_adv = make_adversarial_patch(
                            dpo_batch["chosen_input_ids"], dpo_batch["chosen_attention_mask"],
                            act_cache, train_adversary=False,
                        )

                        optimizer_d.zero_grad(set_to_none=True)
                        optimizer_a.zero_grad(set_to_none=True)

                        # NOTA SULLA MEMORIA: la loss totale e' w_s*loss_s + w_lm*loss_lm +
                        # w_kl*loss_kl. Invece di sommarle e fare UN backward (che terrebbe in
                        # vita i grafi di TRE forward pass contemporaneamente), facciamo il
                        # backward di ciascun termine gia' pesato subito dopo il suo forward:
                        # i gradienti si accumulano in `.grad` e il risultato e' identico, ma
                        # il picco di RAM e' quello di un solo grafo alla volta.
                        w_s, w_lm, w_kl = (
                            float(weights["safety"]), float(weights["lm"]), float(weights["kl"])
                        )
                        monitor.check_memory()
                        with adversarial_patch(model, lora_weights_adv):
                            loss_s = compute_dpo_loss(model, defender_dpo_batch, ref_logps, dpo_util)
                            (w_s * loss_s).backward()

                        # Obiettivo di retain: restare vicini al comportamento originale sulle
                        # istruzioni benigne. Qui la patch NON e' iniettata (siamo fuori dal
                        # blocco `with`): l'utilita' va preservata sul modello "pulito".
                        monitor.check_memory()
                        loss_lm = compute_lm_loss(model, benign_batch)
                        (w_lm * loss_lm).backward()

                        monitor.check_memory()
                        loss_kl = compute_kl_loss(model, ref_model, benign_batch)
                        (w_kl * loss_kl).backward()

                        total_loss_d = w_s * loss_s.item() + w_lm * loss_lm.item() + w_kl * loss_kl.item()

                        grad_norm, skipped = clip_or_skip(defender_params, optimizer_d)
                        if skipped:
                            skipped_steps["defender"] += 1
                            monitor.log_line(
                                f"[defender] e{epoch} b{block_idx} s{step}: gradiente non finito "
                                f"({grad_norm}), passo saltato (#{skipped_steps['defender']})",
                                style="yellow",
                            )

                        monitor.record(
                            phase="defender",
                            epoch=epoch,
                            block=block_idx,
                            step=step,
                            k_steps=k_steps,
                            metrics={
                                "defender_safety_loss": loss_s.item(),
                                "defender_lm_loss": loss_lm.item(),
                                "defender_kl_loss": loss_kl.item(),
                            },
                            extra={"defender_total_loss": total_loss_d, "grad_norm": grad_norm, "skipped": int(skipped)},
                        )
                        if time_budget_exhausted():
                            raise TimeBudgetExceeded(max_minutes)

                    if on_block_end is not None:
                        on_block_end(epoch, block_idx)
    except TimeBudgetExceeded:
        stopped_by_time_budget = True
        monitor.log_line(
            f"\n[run] budget di tempo esaurito ({max_minutes:g} min): fermo il training a fine passo.",
            style="yellow",
        )

    monitor.stopped_by_time_budget = stopped_by_time_budget
    model.eval()
    if skipped_steps["adversary"] or skipped_steps["defender"]:
        monitor.log_line(
            f"[run] passi saltati per gradiente non finito: adversary={skipped_steps['adversary']} "
            f"defender={skipped_steps['defender']}",
            style="yellow",
        )
    monitor.skipped_steps = dict(skipped_steps)
    monitor.log_line(
        "\nImmunizzazione completata. Il modello e' un PeftModel NON fuso: "
        "salvandolo si scrive solo l'adapter LoRA del defender.",
        style="bold green",
    )
    return monitor.history
