"""
Tabella compatta dei trial registrati in `selfsrc/runs/trials.jsonl`.

    python -m selfsrc.trials                 # tutti i trial, in ordine cronologico
    python -m selfsrc.trials --last 10       # solo gli ultimi 10
    python -m selfsrc.trials --sort trr      # ordinati per TRR decrescente
    python -m selfsrc.trials --show t07      # tutti i dettagli di UN trial (JSON)
    python -m selfsrc.trials --best          # il miglior trial secondo lo score (vedi sotto)

Pensato per un agente che fa HPO: una riga per trial, i numeri che servono e basta.

Score (usato da --sort score / --best): TRR penalizzato se l'utilita' peggiora oltre la
tolleranza:  score = trr - 5 * max(0, utility_delta - 0.05).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PROJECT_ROOT

DEFAULT_LEDGER = PROJECT_ROOT / "selfsrc" / "runs" / "trials.jsonl"
UTILITY_TOLERANCE = 0.05
UTILITY_PENALTY = 5.0


def load_ledger(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score(row: Dict[str, Any]) -> Optional[float]:
    trr = row.get("trr")
    if trr is None:
        return None
    ud = row.get("utility_delta") or 0.0
    return trr - UTILITY_PENALTY * max(0.0, ud - UTILITY_TOLERANCE)


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "y" if x else "n"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    return str(x)


def _short_overrides(ov: List[str], width: int = 60) -> str:
    # Togliamo i prefissi piu' lunghi, che non aggiungono informazione in tabella.
    text = " ".join(
        o.replace("training.", "t.").replace("data.", "d.").replace("adversary.", "a.")
         .replace("model.lora.", "lora.").replace("attack_eval.", "att.").replace("dpo.", "")
        for o in ov if not o.startswith("run.name=")
    )
    return text if len(text) <= width else text[: width - 1] + "…"


def render_table(rows: List[Dict[str, Any]]) -> str:
    cols = [
        ("name", 10), ("status", 6), ("score", 6), ("trr", 6),
        ("sm_after", 8), ("pref_after", 8), ("sm_att", 8), ("pref_att", 8),
        ("u_delta", 7), ("skip", 5), ("train", 5), ("total", 5), ("rss", 4), ("overrides", 60),
    ]
    header = " ".join(f"{c:<{w}}" for c, w in cols)
    lines = [header, "-" * len(header)]
    for r in rows:
        a = r.get("after") or {}
        att = r.get("attacked") or {}
        sk = r.get("skipped") or {}
        vals = {
            "name": str(r.get("name"))[:10],
            "status": str(r.get("status", "ok"))[:6],
            "score": _f(score(r)),
            "trr": _f(r.get("trr")),
            "sm_after": _f(a.get("safety_margin")),
            "pref_after": _f(a.get("safe_pref_rate"), 2),
            "sm_att": _f(att.get("safety_margin")),
            "pref_att": _f(att.get("safe_pref_rate"), 2),
            "u_delta": _f(r.get("utility_delta")),
            "skip": f"{sk.get('adversary', 0)}/{sk.get('defender', 0)}" if sk else "-",
            "train": _f(r.get("train_min", r.get("wall_min")), 1),
            "total": _f(r.get("total_min"), 1),
            "rss": _f(r.get("peak_rss_gb"), 1),
            "overrides": _short_overrides(r.get("overrides") or []),
        }
        lines.append(" ".join(f"{vals[c]:<{w}}" for c, w in cols))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Registro dei trial HPO")
    ap.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--sort", choices=["time", "trr", "score", "sm_after", "u_delta"], default="time")
    ap.add_argument("--show", default=None, help="nome del trial da mostrare per intero")
    ap.add_argument("--best", action="store_true", help="mostra solo il trial con lo score migliore")
    args = ap.parse_args(argv)

    rows = load_ledger(Path(args.ledger))
    if not rows:
        print(f"Nessun trial in {args.ledger}")
        return 0

    if args.show:
        found = [r for r in rows if r.get("name") == args.show]
        if not found:
            print(f"Trial '{args.show}' non trovato.")
            return 1
        print(json.dumps(found[-1], indent=2))
        return 0

    if args.best:
        scored = [r for r in rows if score(r) is not None]
        if not scored:
            print("Nessun trial con TRR definito.")
            return 1
        best = max(scored, key=score)
        print(render_table([best]))
        return 0

    if args.sort != "time":
        key = {
            "trr": lambda r: r.get("trr") if r.get("trr") is not None else float("-inf"),
            "score": lambda r: score(r) if score(r) is not None else float("-inf"),
            "sm_after": lambda r: (r.get("after") or {}).get("safety_margin", float("-inf")),
            "u_delta": lambda r: -(r.get("utility_delta") if r.get("utility_delta") is not None else float("inf")),
        }[args.sort]
        rows = sorted(rows, key=key, reverse=True)
    if args.last:
        rows = rows[-args.last:] if args.sort == "time" else rows[: args.last]

    print(render_table(rows))
    print(f"\n{len(rows)} trial · score = trr - {UTILITY_PENALTY:g}*max(0, u_delta - {UTILITY_TOLERANCE})"
          f" · skip = passi saltati adversary/defender · train/total = minuti · rss = picco GB")
    last = rows[-1]
    if last.get("before"):
        b = last["before"]
        print(f"base (pulito) dell'ultimo trial: sm={b.get('safety_margin')} pref={b.get('safe_pref_rate')} "
              f"lm={b.get('benign_lm_loss')} · base sotto attacco: {last.get('base_attacked')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
