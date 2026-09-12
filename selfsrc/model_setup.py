"""
Costruzione dei modelli: defender (PeftModel), tokenizer e adversary hypernetwork.

Tutti i parametri (model_id, cache_dir, iperparametri LoRA, rango dell'adversary...)
arrivano da `config.json`.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .adversary import Adversary
from .config import Config


def load_tokenizer(cfg: Config):
    """Carica il tokenizer e garantisce la presenza di un pad token."""
    m = cfg.model
    tokenizer = AutoTokenizer.from_pretrained(
        m["model_id"],
        cache_dir=str(cfg.resolve_path(m["cache_dir"])),
        trust_remote_code=bool(m.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        # Per i modelli decoder-only si usa comunemente l'EOS come token di padding.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(cfg: Config) -> torch.nn.Module:
    """Carica una copia del modello base (senza adapter LoRA)."""
    m = cfg.model
    return AutoModelForCausalLM.from_pretrained(
        m["model_id"],
        low_cpu_mem_usage=True,
        return_dict=True,
        dtype=cfg.dtype,
        cache_dir=str(cfg.resolve_path(m["cache_dir"])),
        trust_remote_code=bool(m.get("trust_remote_code", True)),
    )


def build_defender(cfg: Config) -> Tuple[PeftModel, List[torch.nn.Parameter]]:
    """
    Costruisce il DEFENDER: il modello base con sopra un adapter LoRA allenabile
    (l'adapter si chiama come `model.adapter_name`, di default "defender").

    Solo i parametri dell'adapter verranno ottimizzati: il modello base resta congelato.
    Ritorna il PeftModel e la lista dei parametri del defender.
    """
    m = cfg.model
    lora = m["lora"]
    adapter_name = m.get("adapter_name", "defender")

    base_model = load_base_model(cfg)

    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["lora_alpha"]),
        lora_dropout=float(lora["lora_dropout"]),
        bias=lora.get("bias", "none"),
        task_type=lora.get("task_type", "CAUSAL_LM"),
        target_modules=list(lora["target_modules"]),
        modules_to_save=list(lora.get("modules_to_save", [])),
    )

    model = PeftModel(base_model, peft_config, adapter_name=adapter_name)

    # Gradient checkpointing: durante il backward le attivazioni dei layer vengono
    # RICALCOLATE invece che tenute in memoria dal forward. Costa ~30% di tempo in piu'
    # ma toglie 2-4 GB dal picco di RAM (misurato: 8.95 -> 6.56 GB per passo con
    # max_length=256, batch 2), con loss e gradienti bit-identici. Il notebook lo aveva
    # tolto come "roba da GPU": su una macchina da 15 GB e' invece indispensabile.
    # NB: transformers lo applica solo in modalita' train() -- vedi immunise.py, Fase 1.
    if bool(m.get("gradient_checkpointing", True)):
        model.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    # I parametri "del defender" sono riconoscibili dal nome: PEFT inserisce il nome
    # dell'adapter nel path di ogni tensore che gli appartiene.
    defender_params = [p for n, p in model.named_parameters() if adapter_name in n]
    if not defender_params:
        raise RuntimeError(
            f"Nessun parametro trovato per l'adapter '{adapter_name}': "
            "controlla target_modules nel config."
        )
    return model, defender_params


class ReferenceModel:
    """
    Il modello di RIFERIMENTO: il comportamento del modello base, SENZA l'adapter del
    defender. Serve alla loss DPO (come baseline delle preferenze) e alla loss KL (come
    comportamento originale da non perdere). E' sempre congelato e valutato in no_grad.

    Due modalita' (config `model.reference_model.mode`):

      - "shared" (default): NESSUNA copia dei pesi. Si usa lo stesso PeftModel del
        defender dentro `model.disable_adapter()`, che bypassa sia i layer LoRA sia le
        copie in `modules_to_save` e restituisce esattamente i logit del modello base
        (verificato: differenza massima 0.0 anche con l'adapter perturbato). Costa
        ~0 byte in piu' invece dei ~2-3 GB di una copia. UNICO VINCOLO: va chiamato
        quando la patch adversariale NON e' iniettata (vedi immunise.py, che calcola
        le log-prob di riferimento PRIMA di applicare la patch).

      - "copy": il comportamento originale del notebook, una `deepcopy` congelata di
        `model.base_model`. Piu' RAM, ma indipendente dallo stato del defender.

    L'oggetto si usa come un modello: `ref(input_ids=..., attention_mask=...)` ritorna
    l'output HF (con `.logits`), cosi' le funzioni di loss non devono sapere quale
    modalita' e' attiva.
    """

    def __init__(self, model: PeftModel, device: torch.device, mode: str = "shared"):
        if mode not in ("shared", "copy"):
            raise ValueError(f"reference_model.mode deve essere 'shared' o 'copy', non '{mode}'")
        self.mode = mode
        self.model = model
        self.copy: Optional[torch.nn.Module] = None
        if mode == "copy":
            with torch.no_grad():
                self.copy = copy.deepcopy(model.base_model)
            # Non ottimizzeremo mai la copia.
            self.copy.eval().to(device)
            for p in self.copy.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        if self.mode == "copy":
            return self.copy(*args, **kwargs)
        # Modalita' condivisa: adapter spento e modalita' eval (niente dropout LoRA)
        # solo per la durata di questo forward, poi si ripristina tutto com'era.
        was_training = self.model.training
        self.model.eval()
        try:
            with self.model.disable_adapter():
                return self.model(*args, **kwargs)
        finally:
            if was_training:
                self.model.train()


def warm_start(cfg: Config, model: PeftModel, adversary: Adversary) -> Dict[str, Optional[str]]:
    """
    Carica, se richiesto dal config, i pesi di una run precedente:
      model.init_adapter_from  -> cartella `defender_lora_adapter/` di una run
      adversary.init_from      -> file `adversary.pt` di una run
    Serve a riprendere una run lunga interrotta, o a continuare il miglior trial HPO con
    piu' passi invece di ripartire da zero. La configurazione LoRA/adversary deve essere
    la stessa della run di origine.
    """
    loaded: Dict[str, Optional[str]] = {"adapter": None, "adversary": None}

    adapter_dir = cfg.model.get("init_adapter_from")
    if adapter_dir:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        path = cfg.resolve_path(adapter_dir)
        adapter_name = cfg.model.get("adapter_name", "defender")
        # PEFT salva un adapter che non si chiama "default" in una SOTTOCARTELLA con il
        # suo nome: accettiamo sia la cartella della run sia la sottocartella diretta.
        candidates = [path / adapter_name / "adapter_model.safetensors", path / "adapter_model.safetensors"]
        file = next((c for c in candidates if c.exists()), None)
        if file is None:
            raise FileNotFoundError(f"Nessun adapter_model.safetensors in {path} (ne' in {path / adapter_name})")
        state = load_file(str(file))
        result = set_peft_model_state_dict(model, state, adapter_name=adapter_name)
        unexpected = getattr(result, "unexpected_keys", [])
        if unexpected:
            raise RuntimeError(f"Chiavi inattese caricando l'adapter da {path}: {list(unexpected)[:5]} ...")
        loaded["adapter"] = str(path)

    adv_file = cfg.adversary.get("init_from")
    if adv_file:
        path = cfg.resolve_path(adv_file)
        payload = torch.load(str(path), map_location="cpu")
        adversary.load_state_dict(payload["state_dict"])
        loaded["adversary"] = str(path)

    return loaded


def build_reference_model(cfg: Config, model: PeftModel, device: torch.device) -> ReferenceModel:
    mode = str(cfg.model.get("reference_model", {}).get("mode", "shared"))
    return ReferenceModel(model, device, mode=mode)


def build_layer_configs(cfg: Config) -> Dict[str, Tuple[int, int]]:
    """
    Determina le forme (in_features, out_features) delle proiezioni di attenzione che
    l'adversary deve saper attaccare.

    Se `adversary.layer_configs.auto` e' true, le deduciamo dalla config del modello
    (cosi' il codice funziona anche cambiando model_id); altrimenti usiamo i valori
    scritti a mano in `adversary.layer_configs.manual`.
    """
    lc = cfg.adversary["layer_configs"]
    if not lc.get("auto", True):
        return {k: tuple(v) for k, v in lc["manual"].items()}

    m = cfg.model
    model_config = AutoConfig.from_pretrained(
        m["model_id"],
        cache_dir=str(cfg.resolve_path(m["cache_dir"])),
        trust_remote_code=bool(m.get("trust_remote_code", True)),
    )

    hidden = int(model_config.hidden_size)
    n_heads = int(model_config.num_attention_heads)
    # Con Grouped-Query Attention le teste key/value sono meno di quelle query: le
    # proiezioni k_proj e v_proj producono quindi un output piu' piccolo di hidden.
    n_kv_heads = int(getattr(model_config, "num_key_value_heads", n_heads) or n_heads)
    head_dim = int(getattr(model_config, "head_dim", None) or hidden // n_heads)
    kv_dim = head_dim * n_kv_heads

    return {
        "q_proj": (hidden, head_dim * n_heads),
        "o_proj": (head_dim * n_heads, hidden),
        "k_proj": (hidden, kv_dim),
        "v_proj": (hidden, kv_dim),
    }


def build_adversary(cfg: Config, layer_configs: Dict[str, Tuple[int, int]]) -> Adversary:
    """Istanzia l'adversary hypernetwork con i parametri del config."""
    a = cfg.adversary
    # Teniamo solo i layer che vogliamo davvero attaccare, secondo il config di training.
    wanted = set(cfg.training["target_modules_to_attack"])
    filtered = {k: v for k, v in layer_configs.items() if k in wanted}
    missing = wanted - set(filtered)
    if missing:
        raise ValueError(
            f"Manca la forma per i layer {sorted(missing)} in adversary.layer_configs."
        )
    return Adversary(
        r=int(a["r"]),
        layer_configs=filtered,
        enc_dim=int(a.get("enc_dim", 1024)),
        num_heads=int(a.get("num_heads", 16)),
        use_body=bool(a.get("use_body", False)),
    )


def describe_models(model: PeftModel, adversary: Adversary, defender_params: List[torch.nn.Parameter]) -> Dict[str, Any]:
    """Riassunto numerico (parametri totali/allenabili) da stampare a inizio run."""
    total = sum(p.numel() for p in model.parameters())
    defender_total = sum(p.numel() for p in defender_params)
    return {
        "model_params_total": total,
        "defender_params_total": defender_total,
        "defender_params_tensors": len(defender_params),
        "defender_params_pct": 100.0 * defender_total / total if total else 0.0,
        "adversary_params_total": adversary.num_parameters(),
    }
