"""
Entrypoint del processo di immunizzazione.

    python -m selfsrc.run                        # usa selfsrc/config.json
    python -m selfsrc.run --config altro.json    # usa un altro config
    python -m selfsrc.run --set training.k_steps=5 --set data.n_harmful=4
    python -m selfsrc.run --name t01 --quiet     # per HPO: niente output a schermo,
                                                 # solo UNA riga JSON di risultato alla fine
    python -m selfsrc.run --dry-run              # solo setup + valutazione iniziale

Tutto quello che serve a seguire la run finisce nella cartella
`selfsrc/runs/<nome>-<timestamp>/`:

    config.used.json        il config esatto usato
    metrics.jsonl           una riga JSON per passo (tailabile durante il training)
    console.log             lo stesso output testuale che vedi a schermo
    live_losses.png         grafico rigenerato ogni `monitoring.plot_every` passi
    training_dynamics.png   grafico finale delle quattro loss
    safety_utility.png      confronto prima/dopo
    history.json            storico completo delle loss
    summary.json            riassunto finale (metriche, tempi, valutazione, attacco)
    defender_lora_adapter/  l'adapter LoRA del defender + tokenizer
    adversary.pt            i pesi dell'hypernetwork allenato

Ogni run aggiunge inoltre UNA riga a `selfsrc/runs/trials.jsonl` (il registro dei trial):
`python -m selfsrc.trials` la mostra in tabella.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .attack import (
    attack_cache_key,
    restore_params,
    run_harmful_finetune_attack,
    snapshot_params,
    tamper_resistance_recovered,
)
from .config import Config, apply_runtime_settings, load_config
from .data import build_dataloaders, build_datasets, check_supervision, suggest_max_length
from .evaluate import compare, evaluate_model, generation_smoke_test, make_eval_dpo_util
from .immunise import run_immunisation, total_planned_steps
from .model_setup import (
    build_adversary,
    build_defender,
    build_layer_configs,
    build_reference_model,
    describe_models,
    load_base_model,
    load_tokenizer,
    warm_start,
)
from .monitor import LowMemoryError, RunMonitor, plot_safety_utility, read_memory_gb


# ---------------------------------------------------------------------------
# Riga di comando
# ---------------------------------------------------------------------------

def _coerce(value: str) -> Any:
    """Converte la stringa di un override CLI nel tipo piu' plausibile (int/float/bool/JSON)."""
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def apply_overrides(cfg: Config, overrides: List[str]) -> None:
    """Applica override `sezione.chiave=valore` presi da riga di comando, sopra al config."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override malformato (serve chiave=valore): {item}")
        dotted, raw_value = item.split("=", 1)
        parts = dotted.split(".")
        node = cfg.raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(raw_value)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Immunizzazione AntiDote (bi-level adversarial training)")
    parser.add_argument("--config", default=None, help="Percorso del config JSON (default: selfsrc/config.json)")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="Override puntuale, es. --set training.k_steps=5 (ripetibile)",
    )
    parser.add_argument("--name", default=None, help="Nome della run (scorciatoia per --set run.name=...)")
    parser.add_argument("--note", default="", help="Nota libera salvata nel registro dei trial")
    parser.add_argument(
        "--quiet", action="store_true",
        help="Niente output a schermo durante la run; alla fine stampa UNA riga JSON con i risultati.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Prepara tutto e valuta il modello iniziale, ma non esegue il training.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Registro dei trial
# ---------------------------------------------------------------------------

def _pick(d: Optional[Dict[str, Any]], keys) -> Optional[Dict[str, Any]]:
    if d is None:
        return None
    return {k: (round(d[k], 5) if isinstance(d[k], float) else d[k]) for k in keys if k in d}


def compact_result(summary: Dict[str, Any], cfg: Config, args: argparse.Namespace) -> Dict[str, Any]:
    """
    La riga che va nel registro dei trial e che `--quiet` stampa a fine run: SOLO i numeri
    che servono a confrontare due trial, niente storici. Pensata per essere letta da un
    agente senza inondare il contesto.
    """
    ev = summary.get("eval", {})
    before = ev.get("Base (originale)")
    after = ev.get("AntiDote (immunizzato)")
    att = summary.get("attack_eval") or {}
    tr = summary.get("training", {})
    keys = ("safety_margin", "safety_margin_se", "safe_pref_rate", "benign_lm_loss")
    tail = {
        k.replace("defender_", "").replace("_loss", ""): round(v["mean_tail"], 4)
        for k, v in tr.items()
        if isinstance(v, dict) and "mean_tail" in v
    }
    return {
        "name": cfg.run.get("name"),
        "run_dir": summary.get("run_dir"),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": args.note,
        "overrides": list(args.overrides),
        "status": summary.get("status", "ok"),
        "steps_done": tr.get("steps_done"),
        "train_min": round(tr.get("wall_time_s", 0) / 60.0, 2),
        "total_min": round(summary.get("total_wall_time_s", 0) / 60.0, 2),
        "peak_rss_gb": tr.get("peak_rss_gb"),
        "skipped": tr.get("skipped_steps"),
        "time_budget_hit": tr.get("stopped_by_time_budget", False),
        "loss_tail": tail,
        "before": _pick(before, keys),
        "after": _pick(after, keys),
        "attacked": _pick(att.get("hardened_attacked"), keys),
        "base_attacked": _pick(att.get("base_attacked"), keys),
        "trr": att.get("trr"),
        "utility_delta": summary.get("eval_delta", {}).get("benign_lm_loss"),
        "safety_delta": summary.get("eval_delta", {}).get("safety_margin"),
    }


def append_to_ledger(cfg: Config, row: Dict[str, Any]) -> Path:
    ledger = cfg.resolve_path(cfg.run.get("output_root", "selfsrc/runs")) / "trials.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as f:
        f.write(json.dumps(row) + "\n")
    return ledger


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    cfg = load_config(args.config)
    if args.name:
        cfg.raw["run"]["name"] = args.name
    apply_overrides(cfg, args.overrides)
    if args.quiet:
        cfg.raw["monitoring"]["use_rich"] = False
        cfg.raw["monitoring"]["quiet"] = True
    apply_runtime_settings(cfg)

    run_dir = cfg.make_run_dir()
    device = cfg.device
    total_steps = total_planned_steps(cfg)
    mon_cfg = cfg.monitoring

    summary: Dict[str, Any] = {"run_dir": str(run_dir), "status": "ok"}
    exit_code = 0
    t_run_start = time.time()

    with RunMonitor(run_dir, mon_cfg, total_steps) as monitor:
        monitor.log_line(f"[run] cartella di output: {run_dir}", style="bold")
        monitor.log_line(f"[run] device={device}  dtype={cfg.dtype}  seed={cfg.run.get('seed')}")

        # --- 0. RAM: ci sta tutto? Meglio saperlo PRIMA di caricare i modelli. ---
        mem = read_memory_gb()
        min_free = float(cfg.run.get("min_available_ram_gb", 0) or 0)
        if mem["available_gb"] is not None:
            monitor.log_line(
                f"[memoria] RAM disponibile all'avvio: {mem['available_gb']:.2f} GB "
                f"(minimo richiesto: {min_free:.2f} GB; stop automatico sotto "
                f"{float(mon_cfg.get('abort_if_available_ram_below_gb', 0) or 0):.2f} GB)"
            )
            if min_free and mem["available_gb"] < min_free:
                msg = (
                    f"RAM disponibile {mem['available_gb']:.2f} GB < run.min_available_ram_gb={min_free:.2f}. "
                    "Chiudi qualcosa o abbassa la soglia nel config."
                )
                monitor.log_line("[memoria] " + msg, style="bold red")
                raise SystemExit(msg)
        monitor.log_line(f"[run] passi pianificati: {total_steps} "
                         f"({cfg.training['epochs']} epoche x {cfg.training['num_blocks']} blocchi "
                         f"x {cfg.training['k_steps']} passi x 2 fasi)")

        # --- 1. Dati (il tokenizer serve prima, per il filtro per lunghezza) ---
        tokenizer = load_tokenizer(cfg)
        datasets = build_datasets(cfg, tokenizer)
        monitor.log_line(
            "[dati] train dannoso={} · train benigno={} · eval dannoso={} · eval benigno={} · attacco={}"
            " · filtro lunghezza={} · holdout={}".format(
                len(datasets["train_harmful"]), len(datasets["train_benign"]),
                len(datasets["eval_harmful"]), len(datasets["eval_benign"]),
                len(datasets["attack_harmful"]),
                cfg.data.get("filter_by_length", False), cfg.data.get("holdout", False),
            )
        )

        # --- 2. Modelli ----------------------------------------------------
        monitor.log_line(f"[modello] carico {cfg.model['model_id']} ...")
        model, defender_params = build_defender(cfg)
        model.to(device)
        # Stato iniziale del defender (LoRA B=0, copie identiche al base): serve per
        # rieseguire l'attacco sul modello NON immunizzato (baseline) senza ricaricarlo.
        initial_state = snapshot_params(defender_params)

        layer_configs = build_layer_configs(cfg)
        monitor.log_line(f"[adversary] forme dei layer bersaglio: {layer_configs}")
        adversary = build_adversary(cfg, layer_configs).to(device)

        loaded = warm_start(cfg, model, adversary)
        if loaded["adapter"] or loaded["adversary"]:
            monitor.log_line(f"[modello] warm start: adapter={loaded['adapter']} adversary={loaded['adversary']}")

        stats = describe_models(model, adversary, defender_params)
        monitor.log_line(
            "[modello] parametri totali={model_params_total:,} · defender allenabili="
            "{defender_params_total:,} ({defender_params_pct:.2f}%) su {defender_params_tensors} tensori · "
            "adversary={adversary_params_total:,}".format(**stats)
        )

        ref_model = build_reference_model(cfg, model, device)
        monitor.log_line(
            f"[modello] modello di riferimento in modalita' '{ref_model.mode}' "
            + ("(pesi condivisi, adapter spento: nessuna copia in RAM)" if ref_model.mode == "shared"
               else "(deepcopy congelata: ~2-3 GB in piu')")
        )

        # --- 2b. Preflight: i batch hanno davvero token su cui calcolare la loss? ---
        supervision = check_supervision(cfg, datasets, tokenizer)
        if supervision["ok"]:
            monitor.log_line(
                "[preflight] ok: con max_length={} tutte le righe hanno token supervisionati.".format(
                    supervision["max_length"]
                )
            )
        else:
            suggested = suggest_max_length(cfg, datasets, tokenizer)
            msg = (
                "[preflight] max_length={} e' troppo corto: {}/{} righe benigne e {}/{} righe dannose "
                "restano SENZA alcun token di risposta dopo la troncatura. La loss LM su quelle righe "
                "vale NaN e il NaN si propaga a tutti i gradienti del defender. "
                "Alza data.max_length (suggerito: >= {}) oppure attiva data.filter_by_length."
            ).format(
                supervision["max_length"],
                supervision["benign"]["rows_without_supervision"], supervision["benign"]["rows_total"],
                supervision["harmful"]["rows_without_supervision"], supervision["harmful"]["rows_total"],
                suggested,
            )
            if cfg.data.get("strict_preflight", True):
                monitor.log_line(msg, style="bold red")
                raise ValueError(msg)
            monitor.log_line(msg + " (strict_preflight=false: proseguo comunque)", style="bold yellow")

        harmful_loader, benign_loader = build_dataloaders(cfg, datasets)
        eval_dpo = make_eval_dpo_util(cfg)

        # --- 3. Valutazione PRIMA -------------------------------------------
        before: Optional[Dict[str, float]] = None
        if cfg.evaluation.get("enabled", True):
            if cfg.evaluation.get("compare_with_pristine", False):
                monitor.log_line("[eval] carico una copia vergine del modello base (riferimento 'prima') ...")
                pristine_model = load_base_model(cfg).to(device).eval()
                for p in pristine_model.parameters():
                    p.requires_grad = False
                before = evaluate_model(pristine_model, datasets, tokenizer, cfg, eval_dpo)
                # Ci serviva solo per questa misura: liberiamo subito i suoi ~2 GB.
                del pristine_model
                gc.collect()
            else:
                # Senza copia vergine misuriamo il defender PRIMA del training: la matrice
                # B di LoRA e' inizializzata a zero e le copie in modules_to_save sono
                # identiche all'originale, quindi i logit sono ESATTAMENTE quelli del
                # modello base (verificato: differenza massima 0.0). Zero RAM in piu'.
                before = evaluate_model(model, datasets, tokenizer, cfg, eval_dpo)
            monitor.log_line(f"[eval] prima: {before}")

        if args.dry_run:
            monitor.log_line("[run] --dry-run: mi fermo prima del training.", style="yellow")
            summary.update({"dry_run": True, "model_stats": stats, "eval_before": before, "preflight": supervision})
            with open(run_dir / mon_cfg.get("summary_name", "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            return 0

        # --- 4. Immunizzazione ---------------------------------------------
        ckpt = cfg.checkpoint

        def save_checkpoint(tag: str = "") -> None:
            """Scrive adapter del defender + pesi dell'adversary nella cartella della run."""
            suffix = f"-{tag}" if tag else ""
            if ckpt.get("save_adapter", True):
                path = run_dir / (ckpt.get("adapter_subdir", "defender_lora_adapter") + suffix)
                model.save_pretrained(str(path))
                tokenizer.save_pretrained(str(path))
            if ckpt.get("save_adversary", True):
                name = ckpt.get("adversary_filename", "adversary.pt")
                if suffix:
                    name = name.replace(".pt", f"{suffix}.pt")
                torch.save(
                    {"state_dict": adversary.state_dict(), "layer_configs": adversary.layer_configs},
                    run_dir / name,
                )

        def on_block_end(epoch: int, block: int) -> None:
            if ckpt.get("save_every_block", False):
                save_checkpoint(f"e{epoch}b{block}")
                monitor.log_line(f"[checkpoint] salvato a fine epoca {epoch} blocco {block}")

        try:
            run_immunisation(
                cfg=cfg, model=model, adversary=adversary, ref_model=ref_model, tokenizer=tokenizer,
                defender_params=defender_params, harmful_dataloader=harmful_loader,
                benign_dataloader=benign_loader, monitor=monitor, on_block_end=on_block_end,
            )
        except KeyboardInterrupt:
            summary["status"] = "interrupted"
            exit_code = 2
            monitor.log_line("\n[run] interrotto dall'utente: salvo lo stato corrente.", style="yellow")
        except LowMemoryError:
            summary["status"] = "low_memory"
            exit_code = 2
            monitor.log_line("\n[run] interrotto per RAM insufficiente: salvo lo stato corrente.", style="bold red")
        gc.collect()

        # --- 5. Grafici e checkpoint ----------------------------------------
        monitor.save_history(run_dir / mon_cfg.get("history_name", "history.json"))
        monitor.plot_losses(run_dir / mon_cfg.get("final_plot_name", "training_dynamics.png"))
        save_checkpoint()
        monitor.log_line(f"[checkpoint] adapter e adversary salvati in {run_dir}")

        summary.update({"model_stats": stats, "training": monitor.summary(), "preflight": supervision})

        # --- 6. Valutazione DOPO --------------------------------------------
        if cfg.evaluation.get("enabled", True):
            after = evaluate_model(model, datasets, tokenizer, cfg, eval_dpo)
            monitor.log_line(f"[eval] dopo:  {after}")
            results = compare(before, after)
            plot_safety_utility(results, run_dir / mon_cfg.get("eval_plot_name", "safety_utility.png"))
            summary["eval"] = results
            if before is not None:
                summary["eval_delta"] = {
                    "safety_margin": after["safety_margin"] - before["safety_margin"],
                    "safe_pref_rate": after["safe_pref_rate"] - before["safe_pref_rate"],
                    "benign_lm_loss": after["benign_lm_loss"] - before["benign_lm_loss"],
                }
                monitor.log_line(
                    "[eval] delta: margine di sicurezza {:+.4f} (se {:.4f}; piu' alto e' meglio) · "
                    "LM loss benigna {:+.4f} (piu' basso e' meglio)".format(
                        summary["eval_delta"]["safety_margin"], after["safety_margin_se"],
                        summary["eval_delta"]["benign_lm_loss"],
                    ),
                    style="bold",
                )

            # --- 7. Attacco di fine-tuning: la vera misura di resistenza al tampering ---
            att_cfg = cfg.attack_eval
            if att_cfg.get("enabled", False) and before is not None:
                hardened_state = snapshot_params(defender_params)
                monitor.log_line(
                    f"[attacco] fine-tuning malevolo sul modello immunizzato: {att_cfg.get('steps')} passi, "
                    f"lr={att_cfg.get('lr')}, harmful_ratio={att_cfg.get('harmful_ratio', 1.0)}, "
                    f"{len(datasets['attack_harmful'])} esempi dannosi / {len(datasets['attack_benign'])} benigni ..."
                )
                trace_h = run_harmful_finetune_attack(cfg, model, defender_params, tokenizer, datasets["attack_harmful"], datasets["attack_benign"], monitor)
                hardened_attacked = evaluate_model(model, datasets, tokenizer, cfg, eval_dpo)
                restore_params(defender_params, hardened_state)
                monitor.log_line(f"[attacco] immunizzato dopo attacco: {hardened_attacked}")

                base_attacked = None
                if att_cfg.get("compare_base", True):
                    cache_dir = cfg.resolve_path(att_cfg.get("cache_dir", "selfsrc/cache"))
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_file = cache_dir / f"base_attack_{attack_cache_key(cfg)}.json"
                    if cache_file.exists():
                        with open(cache_file) as f:
                            base_attacked = json.load(f)["base_attacked"]
                        monitor.log_line(f"[attacco] base dopo attacco (da cache {cache_file.name}): {base_attacked}")
                    else:
                        monitor.log_line("[attacco] stesso attacco sul modello base (una volta sola, poi in cache) ...")
                        restore_params(defender_params, initial_state)
                        trace_b = run_harmful_finetune_attack(cfg, model, defender_params, tokenizer, datasets["attack_harmful"], datasets["attack_benign"], monitor)
                        base_attacked = evaluate_model(model, datasets, tokenizer, cfg, eval_dpo)
                        restore_params(defender_params, hardened_state)
                        with open(cache_file, "w") as f:
                            json.dump({"base_attacked": base_attacked, "attack_losses": trace_b["losses"], "clean": before}, f, indent=2)
                        monitor.log_line(f"[attacco] base dopo attacco: {base_attacked}")

                trr = None
                if base_attacked is not None:
                    trr = tamper_resistance_recovered(
                        before["safety_margin"], base_attacked["safety_margin"], hardened_attacked["safety_margin"]
                    )
                    monitor.log_line(
                        "[attacco] TRR = {} (0 = come il base, 1 = attacco neutralizzato; base: {:.3f} -> {:.3f}, "
                        "immunizzato: {:.3f} -> {:.3f})".format(
                            "n/d (attacco troppo debole: il base non si muove)" if trr is None else f"{trr:.3f}",
                            before["safety_margin"], base_attacked["safety_margin"],
                            after["safety_margin"], hardened_attacked["safety_margin"],
                        ),
                        style="bold",
                    )
                summary["attack_eval"] = {
                    "config": {k: v for k, v in att_cfg.items() if not k.startswith("_")},
                    "hardened_attacked": hardened_attacked,
                    "base_attacked": base_attacked,
                    "trr": trr,
                    "attack_losses_hardened": trace_h["losses"],
                }
                del hardened_state
                gc.collect()

            gen = generation_smoke_test(model, tokenizer, cfg)
            if gen is not None:
                summary["generation_smoke_test"] = gen
                monitor.log_line(f"[gen] prompt: {gen['prompt']}")
                monitor.log_line(f"[gen] risposta: {gen['completion']}")

        summary["total_wall_time_s"] = round(time.time() - t_run_start, 2)
        summary["warm_start"] = loaded
        with open(run_dir / mon_cfg.get("summary_name", "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        row = compact_result(summary, cfg, args)
        ledger = append_to_ledger(cfg, row)
        monitor.log_line(f"[run] riga aggiunta al registro dei trial: {ledger}")
        monitor.log_line(f"\n[run] fatto. Risultati in: {run_dir}", style="bold green")

    if args.quiet:
        # L'UNICA cosa che finisce su stdout in modalita' quiet.
        print(json.dumps(row))
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
