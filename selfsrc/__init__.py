"""
selfsrc -- implementazione autonoma (solo script Python) del processo di
immunizzazione AntiDote.

Questo pacchetto e' una COPIA indipendente del codice necessario: non importa
nulla dai moduli alla root del progetto (adversary.py, loss.py, training.py...),
cosi' puo' essere modificato senza toccarli. Tutti i parametri vivono in
`selfsrc/config.json`.

Entrypoint: `python -m selfsrc.run`
"""

__all__ = [
    "config",
    "activations",
    "adversary",
    "injection",
    "losses",
    "data",
    "model_setup",
    "monitor",
    "immunise",
    "evaluate",
]
