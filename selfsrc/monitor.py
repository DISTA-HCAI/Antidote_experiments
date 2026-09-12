"""
Monitoraggio della run: console, log strutturato e grafici.

L'obiettivo e' che il processo sia FACILE DA SEGUIRE mentre gira:

  - una riga per passo a console, con loss istantanea + media mobile (EMA) + ETA;
  - una barra di avanzamento globale (quanti passi fatti su quanti totali);
  - un file `metrics.jsonl` scritto incrementalmente (una riga JSON per passo): si puo'
    leggere/tailare da un altro terminale mentre il training gira;
  - un `console.log` con la stessa cosa in testo;
  - un grafico `live_losses.png` rigenerato ogni N passi, da tenere aperto in un
    visualizzatore che si auto-aggiorna.

Se la libreria `rich` e' installata (e monitoring.use_rich e' true) usiamo una
dashboard live; altrimenti si ricade su normali `print`, senza perdere nessuna
informazione.
"""

from __future__ import annotations

import gc
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

# matplotlib in modalita' "headless": nessuna finestra, scriviamo solo file PNG.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:  # `rich` e' opzionale: se manca, si va avanti lo stesso.
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False


# Le quattro serie di loss tracciate, con etichetta e colore per i grafici.
LOSS_PANELS = [
    ("adversary_loss", "Adversary DPO loss (Fase 1)", "tab:red"),
    ("defender_safety_loss", "Defender safety DPO loss (Fase 2)", "tab:green"),
    ("defender_lm_loss", "Defender LM retain loss (Fase 2)", "tab:blue"),
    ("defender_kl_loss", "Defender KL retain loss (Fase 2)", "tab:orange"),
]


def _load_malloc_trim():
    """
    Ritorna `malloc_trim` della libc (solo Linux/glibc), oppure None.

    Perche' serve: glibc NON restituisce subito al sistema la memoria liberata dal
    processo (la tiene in "arene" per riusarla), e con tanti tensori medi allocati e
    liberati ad ogni passo il RSS cresce per frammentazione anche se la memoria
    "viva" e' costante. `malloc_trim(0)` forza la restituzione delle pagine libere:
    chiamarlo a fine passo tiene il RSS vicino all'uso reale.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        return libc.malloc_trim
    except (OSError, AttributeError):
        return None


_MALLOC_TRIM = _load_malloc_trim()


def release_memory() -> None:
    """Da chiamare a fine passo: garbage collection + restituzione heap al sistema."""
    gc.collect()
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)


class LowMemoryError(RuntimeError):
    """Sollevata dal monitor quando la RAM disponibile del sistema scende sotto la
    soglia di sicurezza: meglio fermarsi (salvando lo stato) che lasciare che l'OOM
    killer del sistema operativo scelga a caso cosa terminare."""


def peak_rss_gb() -> float:
    """Massimo RSS raggiunto dal processo dall'avvio (high-water mark del kernel), in GB.
    A differenza dei campioni per-passo (letti DOPO il trim), questo cattura anche i
    picchi interni a un passo."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # kB -> GB su Linux
    except (ImportError, OSError):
        return 0.0


def read_memory_gb() -> Dict[str, Optional[float]]:
    """
    Legge RSS del processo e memoria disponibile del sistema, in GB. Solo Linux
    (/proc); altrove ritorna None e il controllo di sicurezza resta disattivato.
    """
    rss = available = None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1e6  # kB -> GB
                    break
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) / 1e6
                    break
    except OSError:
        pass
    return {"rss_gb": rss, "available_gb": available}


class EMA:
    """Media mobile esponenziale: smorza il rumore passo-per-passo e rende leggibile
    la tendenza di una loss che salta molto."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


def fmt_duration(seconds: float) -> str:
    """Formatta una durata in h/m/s, per ETA e tempo trascorso."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class RunMonitor:
    """Raccoglie le metriche, le scrive su disco e le mostra a video."""

    def __init__(self, run_dir: Path, monitoring_cfg: Dict[str, Any], total_steps: int):
        self.run_dir = run_dir
        self.cfg = monitoring_cfg
        self.total_steps = total_steps

        # Soglia di RAM disponibile sotto la quale la run si auto-interrompe (GB).
        # 0 o null = nessun controllo.
        self.abort_below_gb = float(monitoring_cfg.get("abort_if_available_ram_below_gb", 0) or 0)
        self.peak_rss_gb = 0.0
        self.skipped_steps: Dict[str, int] = {}
        self.stopped_by_time_budget = False

        self.log_every = max(1, int(monitoring_cfg.get("log_every", 1)))
        self.plot_every = int(monitoring_cfg.get("plot_every", 0) or 0)
        self.show_progress_bar = bool(monitoring_cfg.get("show_progress_bar", True))

        # Storico completo (in memoria) di ogni serie: finisce in history.json e nei grafici.
        self.history: Dict[str, List[float]] = {key: [] for key, _, _ in LOSS_PANELS}
        self.ema: Dict[str, EMA] = {
            key: EMA(float(monitoring_cfg.get("ema_alpha", 0.2))) for key, _, _ in LOSS_PANELS
        }

        self.steps_done = 0
        self.start_time = time.time()
        # Durata degli ultimi passi: usata per stimare l'ETA sul ritmo recente, non sulla
        # media dall'inizio (piu' reattiva se la velocita' cambia).
        self.recent_step_times: Deque[float] = deque(maxlen=20)
        self._last_step_end = self.start_time

        self.jsonl_path = run_dir / monitoring_cfg.get("jsonl_name", "metrics.jsonl")
        self.console_log_path = run_dir / monitoring_cfg.get("console_log_name", "console.log")
        self.live_plot_path = run_dir / monitoring_cfg.get("live_plot_name", "live_losses.png")

        self._jsonl = open(self.jsonl_path, "a", buffering=1)  # line-buffered: leggibile subito
        self._console_log = open(self.console_log_path, "a", buffering=1)

        # quiet: nulla su stdout (console.log e metrics.jsonl si scrivono comunque).
        self.quiet = bool(monitoring_cfg.get("quiet", False))
        self.use_rich = bool(monitoring_cfg.get("use_rich", True)) and _RICH_AVAILABLE and not self.quiet
        self._console = Console() if self.use_rich else None
        self._progress = None
        self._task_id = None
        self._live = None
        # Ultime righe di stato, mostrate nella dashboard rich.
        self._last_rows: Dict[str, Dict[str, Any]] = {}
        self._phase = "-"
        self._position = "-"

    # --- ciclo di vita ------------------------------------------------------
    def __enter__(self) -> "RunMonitor":
        if self.use_rich and self.show_progress_bar:
            self._progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=None),
                TextColumn("{task.completed}/{task.total} passi"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•ETA"),
                TimeRemainingColumn(),
                console=self._console,
            )
            self._task_id = self._progress.add_task("immunizzazione", total=self.total_steps)
            self._live = Live(self._render(), console=self._console, refresh_per_second=4)
            self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc_val, exc_tb)
        self.plot_losses(self.live_plot_path)
        self._jsonl.close()
        self._console_log.close()

    # --- output -------------------------------------------------------------
    def log_line(self, text: str, style: str = "") -> None:
        """Stampa una riga "fuori banda" (titoli di sezione, avvisi) e la registra nel log."""
        self._console_log.write(text + "\n")
        if self.quiet:
            return
        if self._live is not None:
            self._live.console.print(text, style=style or None)
        elif self._console is not None:
            self._console.print(text, style=style or None)
        else:
            print(text, flush=True)

    def _render(self):
        """Costruisce la dashboard rich: tabella delle metriche + barra di avanzamento."""
        table = Table(expand=True, show_edge=False)
        table.add_column("serie", style="bold")
        table.add_column("ultimo", justify="right")
        table.add_column("EMA", justify="right")
        table.add_column("primo", justify="right")
        table.add_column("delta", justify="right")
        table.add_column("n", justify="right")

        for key, label, _ in LOSS_PANELS:
            values = self.history[key]
            if not values:
                table.add_row(label, "-", "-", "-", "-", "0")
                continue
            first, last = values[0], values[-1]
            delta = last - first
            # Verde se la loss e' scesa rispetto al primo valore, rosso se e' salita.
            delta_style = "green" if delta < 0 else "red"
            ema_value = self.ema[key].value
            table.add_row(
                label,
                f"{last:.4f}",
                f"{ema_value:.4f}" if ema_value is not None else "-",
                f"{first:.4f}",
                f"[{delta_style}]{delta:+.4f}[/{delta_style}]",
                str(len(values)),
            )

        mem = read_memory_gb()
        mem_text = ""
        if mem["rss_gb"] is not None:
            mem_text = (f"   RSS: {mem['rss_gb']:.2f} GB (picco {self.peak_rss_gb:.2f})"
                        f"   RAM libera: {mem['available_gb']:.2f} GB")
        header = f"fase: [bold]{self._phase}[/bold]   posizione: {self._position}   " \
                 f"trascorso: {fmt_duration(time.time() - self.start_time)}{mem_text}"
        parts = [Panel(header, title="AntiDote · immunizzazione", border_style="cyan"), table]
        if self._progress is not None:
            parts.append(self._progress)
        return Group(*parts)

    def set_position(self, phase: str, position: str) -> None:
        """Aggiorna l'intestazione (che fase e in che punto del ciclo siamo)."""
        self._phase = phase
        self._position = position
        if self._live is not None:
            self._live.update(self._render())

    def record(
        self,
        phase: str,
        epoch: int,
        block: int,
        step: int,
        k_steps: int,
        metrics: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Registra un passo di training: aggiorna storico/EMA, scrive una riga JSONL,
        aggiorna la dashboard e (ogni `plot_every`) rigenera il grafico live.
        """
        now = time.time()
        self.recent_step_times.append(now - self._last_step_end)
        self._last_step_end = now
        self.steps_done += 1

        row: Dict[str, Any] = {
            "t": round(now - self.start_time, 3),
            "global_step": self.steps_done,
            "phase": phase,
            "epoch": epoch,
            "block": block,
            "step": step,
            "k_steps": k_steps,
        }

        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
                row[key] = value
                row[f"{key}_ema"] = round(self.ema[key].update(value), 6)
            else:
                row[key] = value

        if extra:
            row.update(extra)

        # Prima restituiamo al sistema la memoria del passo appena concluso, poi misuriamo.
        release_memory()

        # Memoria: finisce nel JSONL e nella dashboard, e fa scattare lo stop di sicurezza.
        mem = read_memory_gb()
        if mem["rss_gb"] is not None:
            self.peak_rss_gb = max(self.peak_rss_gb, mem["rss_gb"], peak_rss_gb())
            row["rss_gb"] = round(mem["rss_gb"], 3)
            row["peak_rss_gb"] = round(self.peak_rss_gb, 3)
            row["ram_available_gb"] = round(mem["available_gb"], 3)

        # 1. Log strutturato, riga per riga: tailabile da un altro terminale.
        self._jsonl.write(json.dumps(row) + "\n")

        self.check_memory(mem)

        # 2. Riga leggibile a console.
        if self.steps_done % self.log_every == 0 or step == k_steps:
            metric_text = "  ".join(
                f"{key.replace('defender_', '').replace('_loss', '')}={value:.4f}"
                f"(ema {self.ema[key].value:.4f})"
                for key, value in metrics.items()
                if key in self.history
            )
            eta = self._eta()
            mem_text = f" | RSS {mem['rss_gb']:.2f}GB libera {mem['available_gb']:.2f}GB" \
                if mem["rss_gb"] is not None else ""
            text = (
                f"[{phase:9s}] e{epoch} b{block} s{step}/{k_steps} "
                f"| {metric_text} | ETA {fmt_duration(eta)}{mem_text}"
            )
            self._console_log.write(text + "\n")
            if not self.use_rich and not self.quiet:
                print(text, flush=True)

        self.set_position(phase, f"epoca {epoch} · blocco {block} · passo {step}/{k_steps}")
        if self._progress is not None:
            self._progress.update(self._task_id, completed=min(self.steps_done, self.total_steps))

        # 3. Grafico live, rigenerato ogni tanto (e' l'operazione piu' costosa qui dentro).
        if self.plot_every and self.steps_done % self.plot_every == 0:
            self.plot_losses(self.live_plot_path)

    def check_memory(self, mem: Optional[Dict[str, Optional[float]]] = None) -> None:
        """Interrompe la run se la RAM disponibile del sistema e' sotto soglia."""
        if not self.abort_below_gb:
            return
        mem = mem or read_memory_gb()
        available = mem.get("available_gb")
        if available is not None and available < self.abort_below_gb:
            msg = (
                f"RAM disponibile {available:.2f} GB < soglia {self.abort_below_gb:.2f} GB "
                f"(RSS di questo processo: {mem.get('rss_gb', 0):.2f} GB). Mi fermo per non "
                "saturare il sistema: riduci max_length/batch_size, togli 'embed_tokens' da "
                "modules_to_save, o alza monitoring.abort_if_available_ram_below_gb."
            )
            self.log_line("[memoria] " + msg, style="bold red")
            raise LowMemoryError(msg)

    def _eta(self) -> float:
        """Stima del tempo rimanente, basata sul ritmo degli ultimi passi."""
        if not self.recent_step_times:
            return 0.0
        avg = sum(self.recent_step_times) / len(self.recent_step_times)
        return avg * max(0, self.total_steps - self.steps_done)

    # --- grafici ------------------------------------------------------------
    def plot_losses(self, path: Path, title: str = "AntiDote · dinamiche del training bi-livello") -> Optional[Path]:
        """Disegna i quattro pannelli di loss (uno per serie) in un unico PNG."""
        if not any(self.history[key] for key, _, _ in LOSS_PANELS):
            return None

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (key, label, color) in zip(axes.flat, LOSS_PANELS):
            values = self.history[key]
            if values:
                ax.plot(range(1, len(values) + 1), values, color=color, linewidth=1.0, alpha=0.55)
                # Sovrapponiamo la EMA: la tendenza si legge molto meglio del dato grezzo.
                ema = EMA(float(self.cfg.get("ema_alpha", 0.2)))
                smoothed = [ema.update(v) for v in values]
                ax.plot(range(1, len(values) + 1), smoothed, color=color, linewidth=2.0, label="EMA")
                ax.legend(loc="best", fontsize=8)
            else:
                ax.text(0.5, 0.5, "nessun dato", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("passo")
            ax.set_ylabel("loss")
            ax.grid(alpha=0.3)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def save_history(self, path: Path) -> Path:
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        return path

    def summary(self) -> Dict[str, Any]:
        """Riassunto finale per serie: primo valore, ultimo, minimo, media della coda."""
        out: Dict[str, Any] = {
            "steps_done": self.steps_done,
            "wall_time_s": round(time.time() - self.start_time, 2),
            "peak_rss_gb": round(max(self.peak_rss_gb, peak_rss_gb()), 3),
            "skipped_steps": self.skipped_steps,
            "stopped_by_time_budget": self.stopped_by_time_budget,
        }
        for key, _, _ in LOSS_PANELS:
            values = self.history[key]
            if not values:
                continue
            tail = values[-max(1, len(values) // 10):]  # ultimo 10% dei passi
            out[key] = {
                "first": values[0],
                "last": values[-1],
                "min": min(values),
                "max": max(values),
                "mean_tail": sum(tail) / len(tail),
                "n": len(values),
            }
        return out


def plot_safety_utility(results: Dict[str, Dict[str, float]], path: Path) -> Path:
    """
    Grafico a barre "prima vs dopo" nello spirito delle Figure 3-4 del paper:
    a sinistra il margine di sicurezza, a destra la loss di utilita'.
    """
    labels = list(results.keys())
    safety_vals = [results[l]["safety_margin"] for l in labels]
    utility_vals = [results[l]["benign_lm_loss"] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(labels, safety_vals, color=["tab:gray", "tab:green"][: len(labels)])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("media logp(sicura) - logp(dannosa)")
    axes[0].set_title(
        "Margine di sicurezza sui prompt dannosi\n(piu' alto = preferisce di piu' la risposta sicura)",
        fontsize=10,
    )

    axes[1].bar(labels, utility_vals, color=["tab:gray", "tab:blue"][: len(labels)])
    axes[1].set_ylabel("LM loss media (nat/token)")
    axes[1].set_title(
        "Utilita' sulle istruzioni benigne\n(piu' basso = piu' vicino al comportamento originale)",
        fontsize=10,
    )

    for ax, values in zip(axes, (safety_vals, utility_vals)):
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
