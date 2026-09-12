"""
Funzioni di costo (copia autonoma delle parti di loss.py effettivamente usate).

Qui vivono le tre loss del processo di immunizzazione:
  - `compute_lm_loss`  : la normale loss di linguaggio (next-token prediction), usata
                         come obiettivo di "retain" del defender sulle istruzioni benigne;
  - `compute_dpo_loss` : la loss DPO (Direct Preference Optimization), usata sia
                         dall'adversary (per preferire la risposta dannosa) sia dal
                         defender (per preferire quella sicura, con orientamento invertito);
  - `compute_kl_loss`  : la divergenza KL rispetto al modello di riferimento congelato,
                         seconda loss di retain del defender.

Rispetto all'originale sono state omesse le funzioni mai usate dal loop
(`log_1_minus_p_loss`, `max_entropy_loss`, `dpo_loss_obj`, ...) e ogni dipendenza da
`accelerate`: qui il processo gira in un singolo processo.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


# ---------------------------------------------------------------------------
# Loss di linguaggio (next-token prediction)
# ---------------------------------------------------------------------------

def log_p_loss(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Calcola la loss di log-probabilita' per un modello di linguaggio.

    Questa funzione calcola la cross-entropy tra i logit predetti e le label vere,
    tipicamente usata nei task di modellazione del linguaggio (cioe': "quanto e'
    probabile, secondo il modello, il prossimo token corretto?").

    Args:
        logits: I logit predetti dal modello, forma (batch_size, sequence_length, vocab_size).
            I "logit" sono i punteggi grezzi (non normalizzati) prima della softmax: piu'
            alto e' il logit di un token, piu' probabile lo ritiene il modello.
        labels: Le label vere, forma (batch_size, sequence_length).
        vocab_size: La dimensione del vocabolario (numero di token possibili).

    Returns:
        La loss calcolata, come tensore scalare (un singolo numero).
    """
    # In un modello autoregressivo, la posizione i-esima dei logit predice il token alla
    # posizione i+1: per questo "shiftiamo" (spostiamo di uno) le label rispetto ai
    # logit. `[..., :-1, :]` scarta l'ultima posizione dei logit (non ha un token
    # "successivo" da confrontare), `[..., 1:]` scarta il primo token delle label (nessun
    # logit lo prevede).
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # CrossEntropyLoss (di default) calcola la media della cross-entropy su tutti i
    # token, ignorando automaticamente quelli con label == -100 (l'`ignore_index` di
    # default): sono esattamente i token di prompt/padding mascherati in data.py.
    loss_fct = CrossEntropyLoss()
    # `CrossEntropyLoss` si aspetta un tensore 2D (N, vocab_size) di logit e un tensore
    # 1D (N,) di label: "appiattiamo" quindi le dimensioni batch e sequenza insieme.
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Serve per il model parallelism: le label devono stare dove stanno i logit.
    shift_labels = shift_labels.to(shift_logits.device)
    return loss_fct(shift_logits, shift_labels)


def _filter_dpo_inputs(inputs: Dict[str, torch.Tensor], chosen: bool = False) -> Dict[str, torch.Tensor]:
    """
    Estrae, da un batch DPO (che contiene sia i tensori "chosen_*" che "rejected_*"),
    solo quelli relativi a UNA delle due risposte, rinominandoli senza prefisso
    (input_ids/attention_mask/labels), cosi' possono essere passati direttamente al
    modello come argomenti "normali".
    """
    prefix = "chosen_" if chosen else "rejected_"
    # Se il batch non ha proprio il formato DPO (niente chiavi con questo prefisso),
    # lo ritorniamo inalterato: permette a questa funzione di essere usata anche su
    # batch "normali" (non DPO) senza rompersi.
    if f"{prefix}input_ids" not in inputs:
        return inputs
    return {
        "input_ids": inputs[f"{prefix}input_ids"],
        "attention_mask": inputs[f"{prefix}attention_mask"],
        "labels": inputs[f"{prefix}labels"],
    }


def _filter_inputs(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Filtra il dizionario di input mantenendo solo le chiavi che un modello Hugging Face
    si aspetta come argomenti (`input_ids`, `attention_mask`, `labels`).
    """
    return {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask", "labels"]}


def resolve_vocab_size(model: torch.nn.Module) -> int:
    """
    Recupera la dimensione del vocabolario in modo robusto.

    Nelle versioni recenti di `transformers` l'attributo `model.vocab_size` non e' piu'
    garantito su tutte le classi (ed e' comunque assente su un `PeftModel`, che inoltra
    gli attributi al modello sottostante solo in parte): ricadiamo su `model.config`.
    """
    vocab_size = getattr(model, "vocab_size", None)
    if isinstance(vocab_size, int):
        return vocab_size
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "vocab_size", None) is not None:
        return int(config.vocab_size)
    raise AttributeError("Impossibile determinare vocab_size dal modello.")


def compute_lm_loss(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
    chosen: bool = False,
) -> torch.Tensor:
    """
    Calcola l'obiettivo standard di "massima probabilita' del prossimo token"
    (next-token prediction), cioe' la normale loss di un modello di linguaggio.

    Supporta sia input "normali" sia input in formato DPO (nel qual caso, tramite
    `chosen`, si sceglie quale delle due risposte usare). Nel loop di immunizzazione
    viene sempre chiamata con un batch "normale" (vedi `preprocess_for_it`) per la loss
    di retain del defender, quindi `chosen` resta al default.
    """
    # `_filter_dpo_inputs` non fa nulla se `inputs` non e' in formato DPO (vedi sopra):
    # qui serve solo a gestire anche il caso DPO in modo uniforme.
    filtered = _filter_dpo_inputs(inputs, chosen)
    outputs = model(**_filter_inputs(filtered), output_hidden_states=False)
    return log_p_loss(outputs.logits, filtered.get("labels"), resolve_vocab_size(model))


# ---------------------------------------------------------------------------
# Loss DPO
# ---------------------------------------------------------------------------

def pad_to_length(
    tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1
) -> torch.Tensor:
    """
    Allunga (fa il padding di) un tensore fino a una lunghezza specificata, lungo una
    dimensione data, riempiendo lo spazio in piu' con `pad_value`.

    Serve a rendere due tensori di lunghezza diversa (es. la sequenza "chosen" e quella
    "rejected" di un batch DPO, che possono avere lunghezze diverse) della stessa
    lunghezza, cosi' da poterli concatenare in un unico tensore (vedi
    `DPOLoss.concatenated_inputs` sotto).
    """
    if tensor.size(dim) >= length:
        # Gia' lungo abbastanza (o troppo): non c'e' nulla da fare.
        return tensor
    # Costruiamo un tensore di "riempimento" della forma giusta (stessa forma
    # dell'originale, tranne che nella dimensione `dim`, dove ha la lunghezza
    # mancante) e lo concateniamo in coda.
    pad_size = list(tensor.shape)
    pad_size[dim] = length - tensor.size(dim)
    return torch.cat(
        [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)],
        dim=dim,
    )


class DPOLoss(torch.nn.Module):
    """
    Modulo di loss per la Direct Preference Optimization (DPO): https://arxiv.org/abs/2305.18290.

    Basato sull'implementazione della libreria TRL di Hugging Face.

    Idea di fondo del DPO: dato un prompt e due risposte (una preferita "chosen", una
    scartata "rejected"), si allena il modello ad aumentare la differenza tra
    "quanto il modello preferisce chosen rispetto a rejected" ORA rispetto a quanto la
    preferiva un modello di riferimento (di solito una copia congelata del modello
    prima del training). In questo modo il modello impara la preferenza SENZA bisogno
    di un vero e proprio modello di reward separato (come servirebbe con RLHF classico).

    Args:
        beta: Parametro di "temperatura" della loss DPO, tipicamente tra 0.1 e 0.5. Piu'
            basso = il modello puo' allontanarsi di piu' dal modello di riferimento.
        label_smoothing: Incertezza sulle etichette (quanto ci fidiamo che "chosen" sia
            davvero sempre la scelta giusta).
        loss_type: Uno tra ['sigmoid', 'hinge', 'ipo', 'kto_pair'].
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
        device: Optional[torch.device] = None,
        average_log_prob: bool = False,
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.device = device
        # False (default, come l'originale): le log-prob di una risposta sono la SOMMA sui
        # suoi token -> con risposte da ~100 token la differenza chosen-rejected vale
        # centinaia e la loss IPO ~1e4-1e5. True: MEDIA per token -> valori O(1), loss
        # O(1), gradienti di scala ragionevole. E' una delle manopole piu' importanti per
        # adattare la ricetta del paper (7B+) a un modello da 0.5B.
        self.average_log_prob = average_log_prob

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Calcola la log-probabilita' delle label date, sotto i logit dati.

        `average_log_prob=False` (il default, ed e' cosi' che viene chiamata da
        `concatenated_forward`) ritorna la SOMMA delle log-probabilita' dei token non
        mascherati, non la media. Se due risposte hanno un numero diverso di token
        "reali", la somma e' influenzata anche dalla lunghezza: e' un dettaglio da tenere
        a mente quando si guardano i valori assoluti di chosen_logps/rejected_logps (per
        il training DPO non cambia nulla di sostanziale, perche' la loss guarda sempre una
        DIFFERENZA calcolata con la stessa convenzione sia per il modello in training che
        per quello di riferimento).

        Returns:
            Un tensore di forma (batch_size,).
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits (dimensioni di batch e sequenza) e labels devono avere la stessa forma."
            )

        if not is_encoder_decoder:
            # Stesso shift autoregressivo visto in log_p_loss: il logit alla posizione i
            # predice il token alla posizione i+1.
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        # Maschera: True dove il token NON e' mascherato (cioe' fa parte della risposta
        # vera, non del prompt ne' del padding).
        loss_mask = labels != label_pad_token_id

        # I token mascherati (-100) non sono indici validi per `torch.gather`: li
        # sostituiamo temporaneamente con 0 (un token "fittizio", il cui valore verra'
        # comunque azzerato dalla maschera qui sotto).
        labels[labels == label_pad_token_id] = 0
        # Per ogni posizione, prendiamo la log-probabilita' del token "label":
        #   log_softmax(logits)[label] = logits[label] - logsumexp(logits).
        # NOTA MEMORIA: l'originale faceva `logits.log_softmax(-1)` e poi il gather, cioe'
        # materializzava (e teneva nel grafo per il backward) una SECONDA copia dei logit
        # (batch x seq x ~152k vocaboli x 4 byte = centinaia di MB). Cosi' invece si
        # tiene solo un vettore (batch x seq) in piu'; il risultato e' identico.
        selected = torch.gather(logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_logps = selected - torch.logsumexp(logits, dim=-1)

        if average_log_prob:
            # Media sui soli token validi.
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        # Somma sui soli token validi (i mascherati contribuiscono 0 grazie a loss_mask).
        return (per_token_logps * loss_mask).sum(-1)

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatena gli input "chosen" e "rejected" in un unico tensore."""
        concatenated_batch: Dict[str, torch.Tensor] = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            # Le sequenze "chosen" e "rejected" possono avere lunghezze diverse (perche'
            # tokenizzate e paddate separatamente in preprocess_for_dpo): prima di
            # concatenarle, dobbiamo portarle tutte alla stessa lunghezza massima.
            max_length = max(
                batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1]
            )

        # Prima passata: gestiamo tutte le chiavi "chosen_*", allungandole (con
        # pad_to_length) fino a max_length e rinominandole in "concatenated_*".
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    # Le label si "paddano" con -100 (valore ignorato dalla loss).
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    # Gli input_ids si paddano con il valore di padding "normale".
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    # L'attention mask si padda con 0 ("non guardare qui").
                    pad_value = 0
                else:
                    continue
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[k], max_length, pad_value=pad_value
                )
        # Seconda passata: stessa cosa per le chiavi "rejected_*", ma questa volta
        # CONCATENANDO (torch.cat lungo dim=0, la dimensione di batch) al risultato
        # "chosen" gia' presente sotto la stessa chiave "concatenated_*": e' qui che le
        # due meta' si uniscono in un unico tensore piu' grande.
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                else:
                    continue
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            # Per i modelli encoder-decoder, il prompt viene passato all'encoder
            # separatamente e va semplicemente duplicato per allinearsi alle due meta'.
            concatenated_batch["concatenated_input_ids"] = (
                batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            )
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    def concatenated_forward(
        self, model: torch.nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Esegue il modello sul batch, concatenando "chosen" e "rejected" in un solo
        forward pass (piu' efficiente di due passaggi separati)."""
        concatenated_batch = self.concatenated_inputs(batch, device=self.device)
        # Quanti esempi "chosen" ci sono: ci servira' per separare di nuovo i risultati.
        len_chosen = batch["chosen_labels"].shape[0]

        # Un UNICO forward pass su [chosen; rejected] concatenati lungo la dimensione di
        # batch: le prime `len_chosen` righe riguardano "chosen", le restanti "rejected".
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
        ).logits

        # Log-probabilita' (somma o media per token, vedi `average_log_prob`) per sequenza.
        all_logps = self.get_batch_logps(
            all_logits, concatenated_batch["concatenated_labels"], average_log_prob=self.average_log_prob
        )

        # Separiamo di nuovo i risultati "chosen" da quelli "rejected".
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcola la loss DPO per un batch di log-probabilita' del modello in training
        ("policy") e del modello di riferimento.

        Returns:
            (losses per-esempio, chosen_rewards, rejected_rewards).
        """
        # "Log-ratio" della policy: quanto il modello in training preferisce chosen
        # rispetto a rejected, in scala logaritmica (differenza di log-probabilita').
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        # Stessa cosa, ma per il modello di riferimento: rappresenta "quanto gia'
        # preferiva chosen" ancora PRIMA di questo training.
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        # La differenza tra le due log-ratio misura QUANTO IN PIU' (rispetto al modello di
        # riferimento) il modello in training preferisce ora chosen rispetto a rejected:
        # e' proprio questa quantita' che le varie loss DPO cercano di spingere in alto.
        logits = pi_logratios - ref_logratios

        if self.loss_type == "sigmoid":
            # Variante originale del paper DPO: massimizza log-sigmoid(beta * logits),
            # con una curva a saturazione che evita di spingere la differenza all'infinito.
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            # Variante "hinge" (come nelle SVM): nessuna penalita' se `logits` supera gia'
            # una soglia (1/beta), penalita' lineare altrimenti.
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # IPO (Identity Preference Optimization): penalizza la DISTANZA di `logits`
            # da un valore obiettivo fisso (1 / (2*beta)), invece di spingerlo
            # all'infinito come fa "sigmoid". E' la variante usata di default in questo
            # progetto: piu' stabile con dataset piccoli/rumorosi.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # Variante ispirata a KTO: usa un termine di "KL" per-batch come riferimento,
            # invece di confrontare direttamente chosen vs rejected accoppiati.
            chosen_kl = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_kl = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_kl)),
                    1 - F.sigmoid(self.beta * (chosen_kl - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Tipo di loss sconosciuto: {self.loss_type}. "
                "Deve essere uno tra ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        # I "reward" impliciti del DPO: quanto, secondo il modello in training, e'
        # cambiata (rispetto al riferimento) la preferenza per ciascuna risposta presa
        # singolarmente. Non partecipano al gradiente (`.detach()`): sono solo metriche
        # diagnostiche.
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards


@torch.no_grad()
def compute_reference_logps(
    ref_model, batch: Dict[str, Union[List, torch.LongTensor]], dpo_loss_util: DPOLoss
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Log-probabilita' (chosen, rejected) del modello di riferimento congelato.

    E' separata da `compute_dpo_loss` per un motivo preciso: con il riferimento in
    modalita' "shared" (vedi model_setup.ReferenceModel) il forward di riferimento deve
    avvenire quando la patch adversariale NON e' iniettata. Il chiamante quindi calcola
    queste log-prob PRIMA di entrare nel blocco `with adversarial_patch(...)`.
    Sono staccate dal grafo: non si propaga mai gradiente attraverso il riferimento.
    """
    reference_chosen_logps, reference_rejected_logps, _, _ = dpo_loss_util.concatenated_forward(
        ref_model, batch
    )
    return reference_chosen_logps.detach(), reference_rejected_logps.detach()


def compute_dpo_loss(
    policy_model: torch.nn.Module,
    batch: Dict[str, Union[List, torch.LongTensor]],
    reference_logps: Tuple[torch.Tensor, torch.Tensor],
    dpo_loss_util: DPOLoss,
) -> torch.Tensor:
    """
    Calcola la loss DPO e la ritorna come tensore differenziabile (cioe' con il grafo
    computazionale ancora collegato, pronto per `.backward()`).
    Questa funzione NON esegue il backward pass: lo fa chi la chiama (vedi immunise.py).

    E' la funzione usata dal loop di immunizzazione sia per la loss dell'adversary
    (Fase 1) sia per la loss di sicurezza del defender (Fase 2) -- con lo stesso identico
    codice, ma passando batch con orientamento chosen/rejected diverso.

    `reference_logps` arriva da `compute_reference_logps`; `dpo_loss_util` dal chiamante,
    cosi' beta/loss_type vengono dal config.json invece di essere hard-coded qui dentro.
    """
    reference_chosen_logps, reference_rejected_logps = reference_logps

    # 1. Log-probabilita' dal modello in training (la "policy", cioe' quello che stiamo
    # effettivamente allenando in questo momento -- puo' essere il defender O il modello
    # con la patch adversariale iniettata, secondo chi chiama questa funzione).
    policy_chosen_logps, policy_rejected_logps, _, _ = dpo_loss_util.concatenated_forward(
        policy_model, batch
    )

    # 2. Calcoliamo le loss per-esempio con il metodo `forward` di DPOLoss.
    losses, _, _ = dpo_loss_util(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )

    # 3. Ritorniamo la media delle loss del batch come singolo tensore scalare.
    return losses.mean()


# ---------------------------------------------------------------------------
# Loss KL (retention)
# ---------------------------------------------------------------------------

def compute_kl_loss(
    model: torch.nn.Module, ref_model: torch.nn.Module, benign_batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Calcola una loss di divergenza KL (approssimata come cross-entropy diretta) tra la
    distribuzione di probabilita' del modello in training e quella del modello di
    riferimento, sugli stessi token benigni. E' una delle due loss di "retain" usate dal
    defender (insieme a `compute_lm_loss`): non basta che il TOKEN PIU' PROBABILE resti lo
    stesso, vogliamo che l'INTERA distribuzione di probabilita' (su tutto il vocabolario)
    non si allontani troppo da quella originale.
    """
    # Passiamo ai due modelli solo le chiavi che si aspettano.
    model_inputs = _filter_inputs(benign_batch)

    # 1. Logit dai due modelli. Il modello di riferimento e' sempre "congelato"
    # (torch.no_grad(), nessun gradiente necessario); il modello in training invece deve
    # mantenere il grafo computazionale, perche' e' lui che va effettivamente allenato.
    # Il riferimento va PRIMA: in modalita' "shared" usa gli stessi pesi del defender con
    # l'adapter spento, e le sue probabilita' vanno calcolate e "congelate" subito, cosi'
    # da liberare i logit di riferimento (grandi: seq x vocabolario) prima del forward
    # della policy.
    with torch.no_grad():
        ref_probs = F.softmax(ref_model(**model_inputs).logits, dim=-1)

    policy_logits = model(**model_inputs).logits

    # 2-3. Cross-entropy "diretta" (forward): -sum(P_riferimento * log(Q_policy)). Questa
    # quantita' e' minima quando le due distribuzioni P e Q coincidono esattamente: e' una
    # buona approssimazione pratica della vera divergenza KL, spesso usata perche' piu'
    # stabile numericamente da calcolare rispetto alla formula KL "esatta".
    #
    # NOTA MEMORIA: l'originale calcolava `F.log_softmax(policy_logits)` per intero, cioe'
    # un'altra copia (batch x seq x ~152k) tenuta nel grafo. Siccome sum(P) = 1 su ogni
    # posizione, vale l'identita'
    #     sum_v P_v * log_softmax(q)_v = sum_v P_v * q_v - logsumexp(q)
    # che da' lo stesso numero senza la copia intera. Il segno negativo e' gestito per
    # convenzione (la loss si minimizza).
    cross_entropy = torch.logsumexp(policy_logits, dim=-1) - (ref_probs * policy_logits).sum(dim=-1)

    # 4. Media sul batch e sulla lunghezza di sequenza, rispettando l'attention mask per
    # non mediare anche sui token di padding (che non sono "vero" contenuto).
    mask = benign_batch["attention_mask"]
    masked_loss = cross_entropy * mask

    # Media solo sui token NON mascherati (padding escluso).
    return masked_loss.sum() / mask.sum()
