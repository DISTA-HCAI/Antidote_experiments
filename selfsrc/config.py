"""
Caricamento e accesso ai parametri di `selfsrc/config.json`.

Regola del progetto: nessun iperparametro hard-coded negli script. Ogni modulo
riceve un oggetto `Config` e legge da li'. Le chiavi che iniziano con "_" sono
solo commenti leggibili dentro al JSON e vengono ignorate.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# La root del progetto e' la cartella che CONTIENE selfsrc/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _strip_comments(node: Any) -> Any:
    """Rimuove ricorsivamente le chiavi di commento (quelle che iniziano con '_')."""
    if isinstance(node, dict):
        return {k: _strip_comments(v) for k, v in node.items() if not k.startswith("_")}
    if isinstance(node, list):
        return [_strip_comments(v) for v in node]
    return node


@dataclass
class Config:
    """Contenitore del config, con qualche comodita' sopra il dizionario grezzo."""

    raw: Dict[str, Any]
    path: Path

    # --- accesso per sezione -------------------------------------------------
    @property
    def run(self) -> Dict[str, Any]:
        return self.raw["run"]

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw["model"]

    @property
    def adversary(self) -> Dict[str, Any]:
        return self.raw["adversary"]

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw["data"]

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw["training"]

    @property
    def monitoring(self) -> Dict[str, Any]:
        return self.raw["monitoring"]

    @property
    def checkpoint(self) -> Dict[str, Any]:
        return self.raw["checkpoint"]

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def attack_eval(self) -> Dict[str, Any]:
        return self.raw.get("attack_eval", {})

    def get(self, dotted: str, default: Any = None) -> Any:
        """Legge un valore annidato con la notazione a punti: `cfg.get('data.batch_size')`."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # --- utilita' derivate ---------------------------------------------------
    def resolve_path(self, relative: str) -> Path:
        """Risolve un percorso del config rispetto alla root del progetto."""
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def device(self) -> torch.device:
        wanted = str(self.run.get("device", "cpu")).lower()
        if wanted == "auto":
            wanted = "cuda" if torch.cuda.is_available() else "cpu"
        if wanted.startswith("cuda") and not torch.cuda.is_available():
            print("[config] CUDA richiesta ma non disponibile: ricado su CPU.")
            wanted = "cpu"
        return torch.device(wanted)

    @property
    def dtype(self) -> torch.dtype:
        name = str(self.run.get("torch_dtype", "float32"))
        if name not in _DTYPES:
            raise ValueError(f"torch_dtype '{name}' non riconosciuto: usa uno tra {list(_DTYPES)}.")
        return _DTYPES[name]

    def make_run_dir(self) -> Path:
        """Crea (e ritorna) la cartella di output di questa run, con timestamp."""
        root = self.resolve_path(self.run.get("output_root", "selfsrc/runs"))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = root / f"{self.run.get('name', 'run')}-{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Teniamo una copia del config effettivamente usato accanto ai risultati:
        # serve a poter riprodurre la run anche dopo aver modificato config.json.
        with open(run_dir / "config.used.json", "w") as f:
            json.dump(self.raw, f, indent=2)
        return run_dir


def load_config(path: Optional[str] = None) -> Config:
    """Carica il config JSON (default: selfsrc/config.json) e ne rimuove i commenti."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config non trovato: {cfg_path}")
    with open(cfg_path) as f:
        raw = json.load(f)
    return Config(raw=_strip_comments(raw), path=cfg_path)


def apply_runtime_settings(cfg: Config) -> None:
    """Applica seed e numero di thread: da chiamare UNA volta, all'avvio."""
    seed = cfg.run.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    threads = int(cfg.run.get("num_threads", 0) or 0)
    if threads > 0:
        torch.set_num_threads(threads)
        # Evita che le librerie BLAS sottostanti creino piu' thread di quanti ne vogliamo.
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
