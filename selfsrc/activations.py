"""
Cattura delle attivazioni interne del modello (copia autonoma di activation_cache.py).

Questa classe serve a "spiare" (senza modificarli) gli input o gli output di certi
sotto-moduli di una rete neurale, mentre questa fa un forward pass normale.
Il meccanismo che PyTorch offre per farlo si chiama "forward hook": e' una funzione
che si registra su un modulo, e che PyTorch chiama automaticamente ogni volta che
quel modulo finisce il suo forward pass, passandole input e output.
In AntiDote, questo serve all'ADVERSARY: per generare una patch LoRA "su misura" per
il prompt corrente, ha bisogno di vedere le attivazioni interne (cioe' i vettori
numerici che scorrono dentro il modello) nei layer che vuole attaccare.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class ActivationCache:
    """Cache di attivazioni basata su forward hook, con gestione esplicita del ciclo di vita."""

    def __init__(
        self,
        model: nn.Module,
        target_modules: List[str],
        reshape: bool = True,
        capture_output: bool = False,
    ):
        # Il modello su cui vogliamo "spiare" le attivazioni.
        self.model = model
        # Nomi "brevi" dei sotto-moduli da osservare (es. "q_proj", "k_proj", ...): ogni
        # modulo del modello il cui nome finale corrisponde a uno di questi verra'
        # osservato (vedi `register_hooks`).
        self.target_modules = target_modules
        # Se True, appiattiamo il tensore catturato da (batch, seq_len, dim) a
        # (batch*seq_len, dim): piu' comodo da passare in seguito a reti che si aspettano
        # un input 2D (vedi come viene usato in adversary.py).
        self.reshape = reshape
        # Se True, catturiamo l'OUTPUT del modulo; se False (default), catturiamo il suo
        # INPUT. In AntiDote vogliamo l'input di q/k/v/o_proj (cioe' cosa "entra" in quella
        # proiezione), quindi il default e' False.
        self.capture_output = capture_output
        # Dizionario {nome_modulo: tensore_catturato}. All'inizio e' vuoto: si popola
        # quando registriamo gli hook (con valore None) e si riempie ad ogni forward pass.
        self.activations: Dict[str, Optional[torch.Tensor]] = {}
        # Lista degli "handle" degli hook registrati: servono per poterli rimuovere in
        # seguito con `remove_hooks()`.
        self.hooks = []
        # Dispositivo (CPU/GPU) su cui si trova il modello: usato per spostare i tensori
        # catturati sullo stesso dispositivo, per evitare errori "tensori su device diversi".
        first_param = next(model.parameters(), None)
        self.device = first_param.device if first_param is not None else torch.device("cpu")

    def _get_hook(self, module_name: str, is_output: bool = False):
        """
        Costruisce (e ritorna) la funzione hook effettiva da registrare su un modulo.
        Usiamo una "closure" (funzione dentro funzione) cosi' ogni hook "ricorda" a quale
        nome di modulo appartiene, senza doverlo passare ogni volta.
        """

        def hook(module, inputs, output):
            # `inputs` e' una tupla di argomenti posizionali passati al modulo: per un
            # `nn.Linear` (come q_proj/k_proj/...) il primo (e unico) argomento e' il
            # tensore in ingresso, quindi prendiamo `inputs[0]`. Se invece vogliamo
            # l'output, PyTorch ce lo passa gia' come tensore singolo.
            tensor = output if is_output else inputs[0]
            if self.reshape:
                # Appiattiamo tutte le dimensioni tranne l'ultima (quella delle feature):
                # da (batch, seq_len, dim) a (batch*seq_len, dim).
                tensor = tensor.reshape(-1, tensor.shape[-1])
            # `.detach()`: stacchiamo il tensore dal grafo computazionale di autograd,
            # perche' questa e' solo una "fotografia" da leggere, non deve propagare
            # gradiente all'indietro attraverso questa cache.
            self.activations[module_name] = tensor.detach().to(self.device)

        return hook

    def register_hooks(self) -> int:
        """
        Cerca dentro il modello tutti i sotto-moduli il cui nome (l'ultimo pezzo, dopo
        l'ultimo punto) corrisponde a uno di `target_modules`, e registra un hook su
        ciascuno di essi. Ritorna quanti hook ha registrato.

        ATTENZIONE: chiamare questo metodo piu' volte SENZA prima chiamare
        `remove_hooks()` registra hook aggiuntivi sopra quelli gia' presenti, invece di
        sostituirli. Per questo il loop di immunizzazione lo chiama UNA volta sola, prima
        del ciclo, e usa `clear_cache()` ad ogni passo.
        """
        registered_count = 0

        # `model.named_modules()` restituisce (nome_completo, modulo) per OGNI
        # sotto-modulo della rete, a qualsiasi profondita' (es.
        # "base_model.model.layers.3.self_attn.q_proj").
        for name, module in self.model.named_modules():
            # Prendiamo solo l'ultimo pezzo del nome (dopo l'ultimo punto), che e' il nome
            # "locale" del modulo (es. "q_proj").
            module_name = name.split(".")[-1]

            # Se questo modulo e' uno di quelli che ci interessano...
            if module_name in self.target_modules:
                hook_fn = self._get_hook(name, self.capture_output)
                # `register_forward_hook` ritorna un "handle": lo salviamo per poter
                # rimuovere l'hook piu' avanti.
                self.hooks.append(module.register_forward_hook(hook_fn))
                # Pre-registriamo la chiave nel dizionario (con valore None), cosi'
                # esiste gia' anche prima del primo forward pass.
                self.activations[name] = None
                registered_count += 1

        return registered_count

    def clear_cache(self) -> None:
        """Svuota solo i VALORI delle attivazioni salvate, mantenendo le chiavi (i nomi
        dei moduli) gia' note. Va chiamato prima di ogni nuovo forward pass di cui si
        vogliono catturare attivazioni "fresche", per non confondere i valori di step
        diversi."""
        for key in self.activations:
            self.activations[key] = None

    def remove_hooks(self) -> None:
        """Rimuove tutti gli hook registrati (chiamando `.remove()` su ciascun handle).
        Va sempre chiamato quando non servono piu', altrimenti gli hook restano attaccati
        al modello per sempre, continuando a "spiare" (e a consumare tempo/memoria) ogni
        futuro forward pass, anche fuori da qualsiasi ciclo di training."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    # Context manager SIMMETRICO: a differenza dell'originale (dove `__exit__` era un
    # no-op e gli hook si accumulavano ad ogni `with`), qui l'uscita dal blocco rimuove
    # davvero gli hook. Usare `with` intorno all'INTERO ciclo di training, mai dentro.
    def __enter__(self) -> "ActivationCache":
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()
