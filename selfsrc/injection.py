"""
Iniezione temporanea della patch LoRA malevola (copia autonoma di peft_injection.py).

Questo file si occupa di "iniettare" temporaneamente, dentro un modello gia' caricato,
una patch di tipo LoRA generata al volo dall'adversary (vedi adversary.py), e di
ripristinare il modello allo stato originale una volta finito.

Richiamo veloce su LoRA (Low-Rank Adaptation): invece di modificare direttamente una
matrice di pesi W (grande, es. 896x896), si impara una correzione a "basso rango"
delta_W = B @ A, dove A ha forma (r, in_features) e B ha forma (out_features, r), con
r molto piu' piccolo di in_features/out_features (qui r=8). Il vantaggio e' che delta_W
ha molti meno parametri da imparare rispetto a W intera, pur potendo rappresentare
correzioni significative. Qui l'adversary genera proprio queste due matrici (che nel
codice si chiamano U e V) "su misura" per ogni prompt, invece di averle fisse.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialPeftWrapper(nn.Module):
    """
    Un modulo "involucro" (wrapper) che si sostituisce temporaneamente a un layer
    originale (es. q_proj) del modello. Quando viene chiamato, calcola PRIMA l'output
    normale del layer originale, POI aggiunge sopra la perturbazione calcolata dalla
    patch LoRA malevola generata dall'adversary. Il risultato e' come se il layer
    originale fosse stato "leggermente alterato" da un attacco, senza pero' aver
    modificato i suoi pesi reali (che restano intatti e recuperabili).
    """

    def __init__(self, original_peft_layer: nn.Module, delta_adv: torch.Tensor):
        super().__init__()
        # Il layer originale (es. q_proj), che continuiamo a usare normalmente.
        self.original_peft_layer = original_peft_layer
        # `delta_adv` e' la matrice di perturbazione gia' "assemblata" (V @ U, vedi sotto
        # in `apply_adversarial_wrappers`), di forma (out_features, in_features), esatta-
        # mente come una matrice di pesi di un layer lineare.
        # La registriamo come "buffer" (non come parametro): un buffer si sposta insieme
        # al modulo tra CPU/GPU/dtype, ma NON viene mai considerato un peso da allenare
        # da questo wrapper (che non deve avere pesi propri allenabili: chi allena questa
        # perturbazione e' l'adversary, altrove). `persistent=False` significa che non
        # viene salvato se si fa `state_dict()`/`save_pretrained()` del modello: e' solo
        # temporanea.
        self.register_buffer("delta_adv", delta_adv, persistent=False)
        # Salviamo le dimensioni del layer originale, utili piu' sotto per i reshape.
        self.in_features = original_peft_layer.in_features
        self.out_features = original_peft_layer.out_features

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 1. Calcoliamo l'output "normale" del layer originale (es. q_proj(x)).
        original_output = self.original_peft_layer(x, *args, **kwargs)

        # 2. Se l'input e' 3D (batch, seq_len, in_features), lo appiattiamo a 2D
        # (batch*seq_len, in_features): piu' semplice da moltiplicare per la matrice di
        # perturbazione.
        original_shape = x.shape
        if x.dim() == 3:
            x_reshaped = x.reshape(-1, original_shape[-1])
        else:
            x_reshaped = x

        # 3. Applichiamo la perturbazione adversariale in 2D.
        # Ci assicuriamo che `delta_adv` sia sullo stesso device e dtype dell'input,
        # altrimenti PyTorch darebbe errore nel moltiplicarli insieme.
        delta_adv_device = self.delta_adv.to(x_reshaped.device, dtype=x_reshaped.dtype)
        delta_adv_device = delta_adv_device.reshape(-1, delta_adv_device.shape[-1])
        # `F.linear(x, W)` calcola x @ W^T: e' la stessa operazione che farebbe un
        # `nn.Linear` con pesi W, ma qui W (`delta_adv_device`) e' generato dinamicamente
        # dall'adversary invece di essere un parametro fisso del modello.
        adversarial_perturbation_2d = F.linear(x_reshaped, delta_adv_device)
        # 4. Riportiamo la perturbazione alla forma 3D originale (batch, seq_len,
        # out_features), cosi' puo' essere sommata all'output normale.
        adversarial_perturbation = adversarial_perturbation_2d.view(
            *original_shape[:-1], self.out_features
        )

        # 5. Sommiamo la perturbazione all'output normale: e' esattamente questo che
        # rende il layer "attaccato" -- il suo output e' diverso da come sarebbe stato
        # senza la patch, ma i pesi originali non sono stati toccati.
        if adversarial_perturbation.shape != original_output.shape:
            raise RuntimeError(
                f"Shape mismatch: perturbazione {adversarial_perturbation.shape}, "
                f"output originale {original_output.shape}"
            )

        return original_output + adversarial_perturbation


def apply_adversarial_wrappers(
    model: nn.Module, lora_weights_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
) -> Dict[str, nn.Module]:
    """
    Sostituisce, dentro `model`, ciascun layer bersaglio (identificato dal nome
    completo, es. "...layers.3.self_attn.q_proj") con un `AdversarialPeftWrapper` che
    include la patch corrispondente generata dall'adversary.

    Args:
        model: il modello (PeftModel) su cui iniettare la patch.
        lora_weights_dict: dizionario {nome_layer: (U_gen, V_gen)}, come prodotto da
            `make_adversarial_patch` in immunise.py. U_gen e V_gen sono le due matrici
            LoRA generate dall'adversary per QUESTO batch/prompt specifico.

    Returns:
        Un dizionario {nome_layer: modulo_originale}, da passare in seguito a
        `restore_original_modules` per rimettere ogni cosa come stava.
    """
    # Qui salviamo, per ogni layer modificato, il modulo ORIGINALE che stiamo per
    # sostituire: ci servira' per il ripristino (vedi `restore_original_modules`).
    original_modules: Dict[str, nn.Module] = {}
    # 'name' e' il percorso completo e corretto verso il modulo (es.
    # "base_model.model.layers.3.self_attn.q_proj").
    for name, (U_gen, V_gen) in lora_weights_dict.items():
        # Troviamo il modulo originale a partire dal suo nome completo.
        original_peft_layer = model.get_submodule(name)
        # Per SOSTITUIRE un sotto-modulo con `setattr`, dobbiamo prima trovare il suo
        # modulo "genitore" (il modulo contenitore immediatamente sopra) e il nome
        # dell'attributo su di esso (l'ultimo pezzo del percorso).
        parent_name = ".".join(name.split(".")[:-1])
        module_name = name.split(".")[-1]
        parent_module = model.get_submodule(parent_name)

        # Moltiplicazione tra matrici a batch: V_gen ha forma (B, out_dim, r) e
        # U_gen ha forma (B, r, in_dim) (B = dimensione di batch, r = rango LoRA).
        # Il risultato `delta_adv` ha forma (B, out_dim, in_dim): e' la matrice di
        # perturbazione "completa" (rango pieno nella forma, anche se costruita da
        # due fattori a basso rango r), pronta per essere usata come i pesi di un
        # layer lineare dentro `AdversarialPeftWrapper`.
        delta_adv = V_gen @ U_gen  # (B, out_dim, r) @ (B, r, in_dim) -> (B, out_dim, in_dim)

        # Controllo di sicurezza: la forma della perturbazione deve corrispondere
        # esattamente a quella dei pesi del layer originale (out_features, in_features).
        expected_shape = (original_peft_layer.out_features, original_peft_layer.in_features)
        delta_adv_shape = (delta_adv.shape[-2], delta_adv.shape[-1])
        if delta_adv_shape != expected_shape:
            # Ripristiniamo quello che abbiamo gia' sostituito, per non lasciare il
            # modello in uno stato a meta' (meta' attaccato, meta' no).
            restore_original_modules(model, original_modules)
            raise ValueError(
                f"Shape mismatch per {name}: delta_adv ha forma {delta_adv.shape}, "
                f"ma il layer si aspetta {expected_shape}."
            )
        # Creiamo il wrapper (che tiene sia il layer originale che la perturbazione)
        # e lo "attacchiamo" al modello nello stesso punto dove stava il layer
        # originale, sovrascrivendo l'attributo sul modulo genitore.
        wrapped_layer = AdversarialPeftWrapper(original_peft_layer, delta_adv)
        setattr(parent_module, module_name, wrapped_layer)
        # Ricordiamoci il modulo originale per poterlo ripristinare piu' avanti.
        original_modules[name] = original_peft_layer

    return original_modules


def restore_original_modules(model: nn.Module, original_modules: Dict[str, nn.Module]) -> None:
    """
    Rimette al loro posto i moduli originali (prima della patch), usando lo stesso
    meccanismo di ricerca "robusta" (tramite nome completo) usato per iniettarli.

    Importante: questa e' una sostituzione STRUTTURALE (un `setattr` sul modulo genitore),
    non una modifica dei tensori in-place. Questo significa che una loss gia' calcolata
    PRIMA di questa chiamata (il cui grafo computazionale fa ancora riferimento al
    wrapper adversariale usato in quel forward pass) resta valida e puo' essere comunque
    "backward-ata" dopo il ripristino: il ripristino cambia solo cosa usera' il PROSSIMO
    forward pass, non il grafo di quelli gia' calcolati.
    """
    for name, original_module in original_modules.items():
        parent_name = ".".join(name.split(".")[:-1])
        module_name = name.split(".")[-1]
        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, module_name, original_module)


@contextmanager
def adversarial_patch(model: nn.Module, lora_weights_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
    """
    Zucchero sintattico: applica la patch all'ingresso del blocco `with` e la rimuove
    SEMPRE all'uscita, anche se dentro al blocco viene sollevata un'eccezione. Cosi'
    non si rischia mai di lasciare il modello permanentemente "attaccato" a causa di un
    errore a meta' passo di training.

        with adversarial_patch(model, lora_weights):
            loss = compute_dpo_loss(...)
    """
    original_modules = apply_adversarial_wrappers(model, lora_weights_dict)
    try:
        yield original_modules
    finally:
        restore_original_modules(model, original_modules)
