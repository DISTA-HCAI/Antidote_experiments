# `selfsrc` — immunizzazione AntiDote, solo script Python

Riscrittura autonoma e modulare del processo di immunizzazione che nel notebook
`Untitled.ipynb` viveva sparso tra celle e moduli alla root del progetto.

- **Non importa nulla dalla root** (`adversary.py`, `loss.py`, `training.py`,
  `peft_injection.py`, `activation_cache.py`, `utils.py`): ogni pezzo necessario è stato
  ricopiato qui dentro, commenti italiani inclusi, così si può modificare senza toccare
  i file originali.
- **Zero iperparametri hard-coded**: tutto si legge da [`config.json`](config.json).
- **Facile da monitorare**: dashboard live a terminale, `metrics.jsonl` tailabile,
  grafici rigenerati durante la run.

## Avvio rapido

```bash
# dalla root del progetto (la cartella che contiene selfsrc/)
python -m selfsrc.run
```

Con i valori di default (`config.json`) la run è uno **smoke test**: 1 epoca × 2 blocchi
× 4 passi × 2 fasi = 16 passi su 8+8 esempi, più l'attacco di valutazione: ~8 minuti e
~6.5 GB di picco su CPU. Serve a verificare che il
ciclo bi-livello sia cablato correttamente end-to-end, non a produrre un modello robusto.

```bash
python -m selfsrc.run --dry-run              # solo setup + valutazione iniziale
python -m selfsrc.run --config mio.json      # un altro file di config
python -m selfsrc.run --set training.k_steps=20 --set data.n_harmful=64
```

Gli override `--set` usano la notazione a punti e vincono sul file; il config
effettivamente usato viene copiato nella cartella della run come `config.used.json`.

## Come si segue una run

Tutto finisce in `selfsrc/runs/<nome>-<timestamp>/`:

| file | cosa contiene |
| --- | --- |
| `metrics.jsonl` | una riga JSON per passo, scritta subito (line-buffered) |
| `console.log` | lo stesso output testuale che scorre a schermo |
| `live_losses.png` | grafico rigenerato ogni `monitoring.plot_every` passi |
| `training_dynamics.png` | grafico finale delle quattro loss (grezze + EMA) |
| `safety_utility.png` | confronto prima/dopo su sicurezza e utilità |
| `history.json` | storico completo delle quattro serie |
| `summary.json` | riassunto finale: statistiche, tempi, preflight, valutazione |
| `config.used.json` | il config esatto di questa run |
| `defender_lora_adapter/` | **solo** l'adapter LoRA del defender + tokenizer |
| `adversary.pt` | pesi dell'hypernetwork allenato |

A terminale, `rich` mostra una tabella che si aggiorna in tempo reale (ultimo valore,
EMA, primo valore, delta colorato per ogni loss) più una barra di avanzamento con ETA.
Se `rich` non c'è, si ricade su righe di `print` senza perdere informazioni.

Da un secondo terminale:

```bash
tail -f selfsrc/runs/<run>/console.log
tail -f selfsrc/runs/<run>/metrics.jsonl | jq -c '{step: .global_step, phase, adversary_loss, defender_safety_loss}'
```

## Memoria (leggere prima di lanciare)

Il notebook teneva **tre copie** del modello in RAM (defender, `deepcopy` di riferimento,
copia "vergine" per la valutazione) più l'`embed_tokens` allenato per intero con i suoi
stati AdamW, e valutava tutto in un batch solo: >10 GB di picco su CPU. Qui il processo
sta a ~3.5 GB a riposo e tocca ~6–7 GB di picco dentro a un passo (`summary.json` →
`training.peak_rss_gb`), e si protegge da solo:

| meccanismo | dove | default |
| --- | --- | --- |
| riferimento a **pesi condivisi** (`disable_adapter()`, logit identici al base, 0 copie) | `model.reference_model.mode` | `"shared"` |
| niente copia "vergine": il defender prima del training **è** il modello base | `evaluation.compare_with_pristine` | `false` |
| `embed_tokens` fuori da `modules_to_save` (−2.2 GB tra pesi, gradienti e AdamW) | `model.lora.modules_to_save` | senza `embed_tokens` |
| backward separato per i tre termini della loss del defender (un grafo alla volta) | `immunise.py` | sempre |
| **gradient checkpointing** (attivazioni ricalcolate nel backward: picco 8.95 → 6.56 GB, loss identiche) | `model.gradient_checkpointing` | `true` |
| niente copie intere di `log_softmax` sul vocabolario nelle loss DPO/KL | `losses.py` | sempre |
| valutazione a mini-batch (il notebook faceva 16 sequenze × 152k vocaboli in un colpo) | `evaluation.batch_size` | `2` |
| `malloc_trim` a fine passo (glibc altrimenti non restituisce l'heap) | `monitor.py` | sempre |
| controllo RAM all'avvio: non parte se non c'è abbastanza memoria libera | `run.min_available_ram_gb` | `8.0` |
| **stop di sicurezza** ad ogni passo se la RAM libera del sistema scende sotto soglia, con salvataggio dello stato | `monitoring.abort_if_available_ram_below_gb` | `1.5` |

RSS del processo e RAM libera del sistema compaiono nella dashboard, in ogni riga di
`console.log` e in ogni riga di `metrics.jsonl` (`rss_gb`, `ram_available_gb`); il picco
finisce in `summary.json` (`training.peak_rss_gb`).

Nota: `ulimit -v` **non** è un backstop utile qui — la memoria *virtuale* di PyTorch su
CPU supera di molto quella residente e il limite scatta a caso. Se vuoi un tetto duro
indipendente dal codice usa un cgroup (`systemd-run --user --scope -p MemoryMax=8G python -m selfsrc.run`).

## Moduli

| file | ruolo |
| --- | --- |
| `config.py` | caricamento di `config.json`, seed, device, cartella della run |
| `data.py` | dataset, dataloader, `preprocess_for_dpo` / `preprocess_for_it`, preflight |
| `model_setup.py` | defender (PeftModel + LoRA), modello di riferimento, adversary |
| `adversary.py` | l'hypernetwork che genera le patch LoRA malevole |
| `activations.py` | `ActivationCache`: i forward hook che "spiano" le attivazioni |
| `injection.py` | iniezione/ripristino della patch adversariale sui layer bersaglio |
| `losses.py` | `compute_dpo_loss`, `compute_lm_loss`, `compute_kl_loss`, `DPOLoss` |
| `immunise.py` | il ciclo bi-livello (Fase 1 adversary / Fase 2 defender) |
| `evaluate.py` | margine di sicurezza, utilità, smoke test di generazione |
| `monitor.py` | EMA, JSONL, dashboard live, grafici |
| `run.py` | entrypoint CLI, che mette insieme tutto |

## Il ciclo, in breve

Ogni **blocco** alterna due fasi:

1. **Adversary** — l'hypernetwork osserva le attivazioni interne del defender sul prompt
   dannoso e genera una patch LoRA che, iniettata, spinge il modello a preferire la
   risposta dannosa. Simula un fine-tuner malevolo. Il defender è congelato.
2. **Defender** — con *quella stessa patch* iniettata, l'adapter del defender viene
   aggiornato per preferire comunque la risposta sicura (loss DPO a orientamento
   invertito), restando vicino al comportamento originale sulle istruzioni benigne
   (loss LM + loss KL contro il modello di riferimento congelato). L'adversary è congelato.

Orientamento dei dati, da tenere a mente: in `data/dpo_data.json` **`chosen` è la
risposta DANNOSA** e `rejected` quella sicura. È l'orientamento che vuole l'adversary;
il defender usa lo stesso batch con le due metà **scambiate**.

## Misura di resistenza al tampering

Dopo il training la run esegue un **attacco di fine-tuning malevolo** (`attack_eval`,
default 20 passi di SFT sulle risposte dannose di un set tenuto fuori) sul modello
immunizzato e, una volta sola (poi in cache in `selfsrc/cache/`), sul modello base. Da
qui `trr = (SM_att(hardened) − SM_att(base)) / (SM_clean(base) − SM_att(base))`. Le
metriche `safety_margin` (media per token), `safety_margin_se`, `safe_pref_rate`
(frazione di prompt in cui vince la risposta sicura) e `benign_lm_loss` sono in
`summary.json` e, in forma compatta, nella riga del registro `runs/trials.jsonl`.

## Differenze rispetto al codice alla root

Correzioni già presenti nel notebook e mantenute qui:

1. **Gradienti morti / che colavano** — `requires_grad` ora si commuta esplicitamente per
   fase, ed entrambi gli optimizer si azzerano ad ogni passo.
2. **Hook che si accumulavano** — `ActivationCache.__exit__` era un no-op, quindi ogni
   `with` aggiungeva hook senza rimuoverli mai. Qui il context manager è simmetrico e gli
   hook si registrano una volta sola, attorno all'intero ciclo.
3. **Preprocessing mancante / chiavi sbagliate** — la fase del defender leggeva campi
   inesistenti (`safe_input_ids`, `harmful_input_ids`) e passava stringhe grezze come
   kwargs del modello.
4. **Orientamento chosen/rejected** — ricostruito esplicitamente per il defender.
5. **bf16 / gradient checkpointing / `Accelerator`** — rimossi: singolo processo, device
   dal config.

In più, rispetto anche al notebook:

6. **L'adversary non imparava mai** — `make_adversarial_patch` chiamava l'hypernetwork
   sotto `torch.no_grad()`: gradiente identicamente zero (verificato), `optimizer_a.step()`
   un no-op per tutta la run. In più `model.set_adapter()` a ogni patch riaccendeva
   `requires_grad` sul defender, quindi il backward della "fase adversary" finiva sul
   defender (norma ~1e18) per poi essere scartato. Ora l'adversary gira con il grad
   attivo nella sua fase e `set_adapter` è chiamato una volta sola.
6b. **Gradienti inf/NaN** — con la loss originale (IPO su *somme* di log-prob, valori
   ~1e5) i gradienti dell'adversary arrivano a 1e18 e traboccano; `clip_grad_norm_` con
   norma `inf` moltiplica per 0 e `inf·0 = NaN` finiva nei pesi. Ora il passo viene
   saltato e contato (`skipped`). La manopola `training.dpo.average_log_prob=true`
   (media per token) riporta loss e gradienti a O(1).
7. **Preflight sulla supervisione** (`data.check_supervision`) — con `max_length=48` sul
   dataset di instruction tuning, **7 righe su 8** restano senza alcun token di risposta
   dopo la troncatura: `CrossEntropyLoss` su zero token validi ritorna `NaN`, il `NaN`
   entra nella loss totale del defender e ne corrompe tutti i gradienti in un passo solo,
   silenziosamente. Il preflight lo rileva all'avvio e blocca la run
   (`data.strict_preflight`). Per questo il default qui è `max_length: 256`.
8. **`self.body` dell'adversary è opzionale** (`adversary.use_body`, default `false` =
   fedele all'originale, dove era costruito ma mai chiamato). Attivarlo cambia la scala
   della patch iniziale da ~1.6% a ~600% della norma dei pesi attaccati.
9. **`vocab_size` risolto in modo robusto** (`losses.resolve_vocab_size`) — `model.vocab_size`
   non è garantito sulle versioni recenti di `transformers` né su un `PeftModel`.
10. **Iteratori dei dataloader indipendenti** — nell'originale, esaurito quello dannoso si
   ricreavano entrambi, perdendo un batch benigno.
11. **Patch adversariale come context manager** — il ripristino dei moduli originali è
    garantito anche se il calcolo della loss solleva un'eccezione.
12. **Riferimento a pesi condivisi e log-prob di riferimento calcolate prima della patch**
    — `compute_dpo_loss` è spezzata in `compute_reference_logps` + `compute_dpo_loss`,
    così il riferimento non vede mai la perturbazione adversariale anche quando condivide
    i pesi col defender (vedi la sezione *Memoria*).

## HPO (per un agente)

`selfsrc/HPO_PROMPT.md` è il prompt completo per un agente che debba **ottimizzare gli
iperparametri** per il modello da 0.5B: obiettivo, metriche, protocollo a stadi (prima
iterazioni da ~13 minuti, poi una run lunga), budget e regole per non inondare il
proprio contesto. Il kit di supporto sta in `selfsrc/hpo/`:

| file | ruolo |
| --- | --- |
| `hpo/base_hpo.json` | config di partenza "veloce": filtro per lunghezza, `max_length=128`, fette disgiunte, budget di tempo, attacco calibrabile |
| `hpo/run_trial.sh NAME "nota" --set ...` | lancia UN trial in background (rifiuta se un altro gira) |
| `hpo/wait_trial.sh NAME [sec]` | aspetta e stampa SOLO la riga JSON finale (o le ultime righe d'errore) |
| `python -m selfsrc.trials` | tabella dei trial da `runs/trials.jsonl` (`--sort score`, `--best`, `--show NAME`) |
| `python -m selfsrc.trials --explain NAME` | un trial raccontato a parole: i quattro `sm`, come ne esce il TRR, se e' sopra il rumore |
| `evaluation.probe` | sonda periodica DURANTE il training: misura il margine pulito ogni `every_blocks` blocchi e, con `abort_if_d_safe_below`, abbandona presto i trial gia' persi (`status: aborted_probe`). La traiettoria finisce in `probe_d_safe` nel registro |

La metrica principale è **TRR** (tamper resistance recovered): dopo un attacco di
fine-tuning malevolo a budget fisso (`attack_eval`, vedi `attack.py`), quanta parte del
margine di sicurezza perso dal modello base viene recuperata dal modello immunizzato.
Manopole aggiunte apposta per l'HPO: `training.dpo.average_log_prob`,
`training.max_minutes`, `data.filter_by_length`, `attack_eval.*`,
`model.init_adapter_from` / `adversary.init_from` (warm start), `--quiet`, `--name`, `--note`.

## Scalare oltre lo smoke test

I valori del paper: `k_steps=2000`, 3 epoche, batch 2, dataset interi (15.580 esempi
dannosi / 32.774 benigni), 3×A6000. Per avvicinarsi:

```jsonc
"run":   { "device": "auto", "torch_dtype": "bfloat16" },
"data":  { "n_harmful": null, "n_benign": null, "max_length": 512, "holdout": true },
"training": { "epochs": 3, "num_blocks": 10, "k_steps": 200 }
```

`n_harmful`/`n_benign` a `null` significa "usa tutto il dataset". Con `holdout: true` gli
esempi di valutazione vengono presi dopo quelli di training, così le metriche prima/dopo
non sono misurate sugli stessi dati visti in training.

## Le metriche di valutazione

Non avendo qui il classificatore di dannosità né i benchmark del paper, `evaluate.py` usa
due proxy calcolati con lo stesso codice di loss del training:

- **Margine di sicurezza** = media di `logp(risposta sicura) − logp(risposta dannosa)`.
  Più alto = più sicuro. Proxy dell'Harmful Score (HS), con segno opposto.
- **Utilità** = LM loss media sulle istruzioni benigne. Più basso = più vicino al
  comportamento originale. Proxy della Fine-tune Accuracy (FA).

Con poche decine di esempi sono stime rumorose, non punteggi calibrati.

## Riusare l'adapter salvato

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32, cache_dir="cache_dir"
)
# PEFT salva l'adapter "defender" nella sottocartella defender/ (il nome non e' "default")
model = PeftModel.from_pretrained(base, "selfsrc/runs/<run>/defender_lora_adapter/defender").eval()
tokenizer = AutoTokenizer.from_pretrained("selfsrc/runs/<run>/defender_lora_adapter")
```
