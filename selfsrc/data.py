"""
Caricamento dataset e preprocessing (copia autonoma di utils.py / delle celle del notebook).

Due formati di dato, due funzioni di preprocessing:

  - DPO ("preference"): {prompt, chosen, rejected}. ATTENZIONE all'orientamento usato in
    `data/dpo_data.json`: 'chosen' e' la risposta DANNOSA e 'rejected' quella SICURA
    (vedi dataset_generation.py alla root). E' esattamente l'orientamento che vuole
    l'ADVERSARY; il DEFENDER usa lo stesso batch con chosen/rejected SCAMBIATI (vedi
    immunise.py).

  - IT ("instruction tuning"): {prompt, response}. Usato per le loss di retain del
    defender (LM + KL) sulle istruzioni benigne.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config


class JsonListDataset(Dataset):
    """
    Dataset minimale su una lista di dizionari letta da un file JSON.

    Volutamente NON usiamo `datasets.Dataset` di Hugging Face: qui serve solo indicizzare
    una lista in memoria, e il `DataLoader` di PyTorch con `collate_fn` ci da' gia' batch
    nella forma {chiave: [lista di stringhe]} che il preprocessing si aspetta.
    """

    def __init__(self, rows: List[Dict[str, Any]], required_keys: Tuple[str, ...]):
        missing = [k for k in required_keys if rows and k not in rows[0]]
        if missing:
            raise ValueError(f"Il dataset non contiene le chiavi richieste: {missing}")
        self.rows = rows
        self.required_keys = required_keys

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.rows[idx]
        return {k: row[k] for k in self.required_keys}

    def as_batch(self) -> Dict[str, List[str]]:
        """Ritorna l'INTERO dataset come un unico batch {chiave: [valori]}: comodo per
        la valutazione su subset piccoli."""
        return {k: [row[k] for row in self.rows] for k in self.required_keys}


def collate_dicts(samples: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """Trasforma una lista di esempi in un dizionario di liste (il formato "batched"
    che si aspettano `preprocess_for_dpo` / `preprocess_for_it`)."""
    if not samples:
        return {}
    return {key: [s[key] for s in samples] for key in samples[0]}


def load_json_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def _slice(rows: List[Dict[str, Any]], n: Optional[int], offset: int = 0) -> List[Dict[str, Any]]:
    """Prende `n` righe a partire da `offset`. `n=None` significa "tutte le restanti"."""
    if n is None:
        return rows[offset:]
    return rows[offset : offset + n]


def _fits(tokenizer, prompt: str, response: str, max_length: int, min_response_tokens: int) -> bool:
    """
    True se, dopo chat template + troncatura a `max_length`, restano almeno
    `min_response_tokens` token di RISPOSTA (cioe' non mascherati) su cui calcolare la
    loss. E' il criterio del filtro per lunghezza (vedi `build_datasets`).
    """
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False,
    )
    prompt_len = len(tokenizer(prompt_text)["input_ids"])
    full_len = len(tokenizer(full_text)["input_ids"])
    return min(full_len, max_length) - prompt_len >= min_response_tokens


def _take(rows: List[Dict[str, Any]], n: Optional[int], start: int, keep) -> Tuple[List[Dict[str, Any]], int]:
    """
    Prende (al massimo) `n` righe che soddisfano `keep`, scandendo `rows` da `start`.
    Ritorna (righe scelte, indice della prima riga NON ancora esaminata): cosi' le fette
    successive (eval, attacco) partono da dove si e' fermata la precedente e restano
    disgiunte anche con il filtro attivo. `n=None` = tutte le righe restanti.
    """
    out: List[Dict[str, Any]] = []
    i = start
    while i < len(rows) and (n is None or len(out) < n):
        if keep(rows[i]):
            out.append(rows[i])
        i += 1
    return out, i


def build_datasets(cfg: Config, tokenizer=None) -> Dict[str, JsonListDataset]:
    """
    Costruisce i dataset usati dal processo, secondo la sezione "data" del config:

      train_harmful / train_benign   -> training bi-livello
      eval_harmful  / eval_benign    -> valutazione prima/dopo (margine di sicurezza, utilita')
      attack_harmful / attack_benign -> esempi per l'attacco di fine-tuning malevolo (vedi attack.py)

    Con `holdout: true` le fette sono DISGIUNTE e consecutive: prima il training, poi
    la valutazione, poi l'attacco. Con `holdout: false` valutazione e attacco ripartono
    dalla riga 0 (ok solo per uno smoke test, non per misurare qualcosa).

    Con `filter_by_length: true` (serve il tokenizer) si tengono solo le righe che, con
    il `max_length` scelto, conservano almeno `min_response_tokens` token di risposta.
    E' LA leva per iterare in fretta: con max_length=128 e senza filtro meta' delle righe
    benigne sono di soli -100 (loss NaN, vedi `check_supervision`); con il filtro si
    puo' scendere a 128 (o 96) token e ogni passo costa la meta'.
    """
    d = cfg.data
    harmful_rows = load_json_rows(cfg.resolve_path(d["harmful_path"]))
    benign_rows = load_json_rows(cfg.resolve_path(d["benign_path"]))

    max_length = int(d["max_length"])
    min_resp = int(d.get("min_response_tokens", 8))
    use_filter = bool(d.get("filter_by_length", False))
    if use_filter and tokenizer is None:
        raise ValueError("data.filter_by_length=true richiede il tokenizer: build_datasets(cfg, tokenizer)")

    if use_filter:
        # Per le coppie di preferenza servono abbastanza token in ENTRAMBE le risposte.
        keep_harmful = lambda r: (  # noqa: E731
            _fits(tokenizer, r["prompt"], r["chosen"], max_length, min_resp)
            and _fits(tokenizer, r["prompt"], r["rejected"], max_length, min_resp)
        )
        keep_benign = lambda r: _fits(tokenizer, r["prompt"], r["response"], max_length, min_resp)  # noqa: E731
    else:
        keep_harmful = keep_benign = lambda r: True  # noqa: E731

    train_harmful, h_next = _take(harmful_rows, d.get("n_harmful"), 0, keep_harmful)
    train_benign, b_next = _take(benign_rows, d.get("n_benign"), 0, keep_benign)

    holdout = bool(d.get("holdout", False))
    eval_harmful, h_next = _take(harmful_rows, d.get("eval_n_harmful"), h_next if holdout else 0, keep_harmful)
    eval_benign, b_next = _take(benign_rows, d.get("eval_n_benign"), b_next if holdout else 0, keep_benign)
    attack_harmful, _ = _take(harmful_rows, d.get("attack_n_harmful", 16), h_next if holdout else 0, keep_harmful)
    # Fetta benigna per l'attacco "misto" (paper: 20% dannoso / 80% benigno), vedi attack_eval.harmful_ratio.
    attack_benign, _ = _take(benign_rows, d.get("attack_n_benign", 32), b_next if holdout else 0, keep_benign)

    return {
        "train_harmful": JsonListDataset(train_harmful, ("prompt", "chosen", "rejected")),
        "train_benign": JsonListDataset(train_benign, ("prompt", "response")),
        "eval_harmful": JsonListDataset(eval_harmful, ("prompt", "chosen", "rejected")),
        "eval_benign": JsonListDataset(eval_benign, ("prompt", "response")),
        "attack_harmful": JsonListDataset(attack_harmful, ("prompt", "chosen", "rejected")),
        "attack_benign": JsonListDataset(attack_benign, ("prompt", "response")),
    }


def build_dataloaders(cfg: Config, datasets: Dict[str, JsonListDataset]) -> Tuple[DataLoader, DataLoader]:
    """Crea i due DataLoader di training (dannoso e benigno)."""
    batch_size = int(cfg.data["batch_size"])
    shuffle = bool(cfg.data.get("shuffle", True))
    harmful_loader = DataLoader(
        datasets["train_harmful"], batch_size=batch_size, shuffle=shuffle, collate_fn=collate_dicts
    )
    benign_loader = DataLoader(
        datasets["train_benign"], batch_size=batch_size, shuffle=shuffle, collate_fn=collate_dicts
    )
    return harmful_loader, benign_loader


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_dpo(
    examples: Dict[str, List[str]],
    tokenizer,
    max_length: int = 1024,
    label_pad_token_id: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    Prepara un batch di dati di preferenza per il training DPO.

    Prende un dizionario di liste (le chiavi 'prompt', 'chosen', 'rejected'), applica il
    chat template, tokenizza e produce i sei tensori richiesti dalla loss DPO.

    Le label sono una copia degli input_ids con il PROMPT mascherato a -100: la loss deve
    guardare solo la risposta, non la domanda (che il modello non deve "imparare a
    generare").
    """
    # Ci assicuriamo che il tokenizer abbia un token di padding; per i modelli
    # decoder-only spesso coincide con l'EOS.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = len(examples["prompt"])

    # --- 1. Formattiamo chosen/rejected con il chat template ---
    chosen_full_texts: List[str] = []
    rejected_full_texts: List[str] = []
    prompt_only_texts: List[str] = []

    for i in range(batch_size):
        prompt_text = examples["prompt"][i]
        chosen_text = examples["chosen"][i]
        rejected_text = examples["rejected"][i]

        messages_chosen = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": chosen_text},
        ]
        chosen_full_texts.append(tokenizer.apply_chat_template(messages_chosen, tokenize=False))

        messages_rejected = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": rejected_text},
        ]
        rejected_full_texts.append(tokenizer.apply_chat_template(messages_rejected, tokenize=False))

        # Solo il prompt: ci serve la sua lunghezza in token per sapere quanto mascherare.
        messages_prompt = [{"role": "user", "content": prompt_text}]
        prompt_only_texts.append(
            tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
        )

    # --- 2. Tokenizziamo tutti i testi formattati ---
    tokenized_chosen = tokenizer(
        chosen_full_texts,
        truncation=True,
        max_length=max_length,
        padding="longest",  # Padding alla sequenza piu' lunga DI QUESTO batch.
    )
    tokenized_rejected = tokenizer(
        rejected_full_texts,
        truncation=True,
        max_length=max_length,
        padding="longest",
    )
    # I prompt li tokenizziamo senza padding, perche' ci interessa solo la lunghezza vera.
    tokenized_prompts = tokenizer(prompt_only_texts, truncation=True, max_length=max_length)

    # --- 3. Costruiamo le label mascherando la parte di prompt ---
    chosen_labels = torch.tensor(tokenized_chosen["input_ids"]).clone()
    rejected_labels = torch.tensor(tokenized_rejected["input_ids"]).clone()

    for i in range(batch_size):
        prompt_len = len(tokenized_prompts["input_ids"][i])

        chosen_labels[i, :prompt_len] = label_pad_token_id
        # Mascheriamo anche i token di padding eventualmente presenti nella risposta.
        chosen_labels[i][torch.tensor(tokenized_chosen["attention_mask"][i]) == 0] = label_pad_token_id

        rejected_labels[i, :prompt_len] = label_pad_token_id
        rejected_labels[i][torch.tensor(tokenized_rejected["attention_mask"][i]) == 0] = label_pad_token_id

    return {
        "chosen_input_ids": torch.tensor(tokenized_chosen["input_ids"]),
        "chosen_attention_mask": torch.tensor(tokenized_chosen["attention_mask"]),
        "chosen_labels": chosen_labels,
        "rejected_input_ids": torch.tensor(tokenized_rejected["input_ids"]),
        "rejected_attention_mask": torch.tensor(tokenized_rejected["attention_mask"]),
        "rejected_labels": rejected_labels,
    }


def preprocess_for_it(
    examples: Dict[str, List[str]],
    tokenizer,
    max_length: int = 1024,
    label_pad_token_id: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    Prepara un batch di dati di instruction tuning per la loss di retain.

    Prende un dizionario di liste (chiavi 'prompt' e 'response') e produce i tre tensori
    (input_ids, attention_mask, labels) attesi da `compute_lm_loss` / `compute_kl_loss`.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = len(examples["prompt"])
    full_texts: List[str] = []
    prompt_only_texts: List[str] = []

    for prompt, response in zip(examples["prompt"], examples["response"]):
        messages_complete = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        full_texts.append(tokenizer.apply_chat_template(messages_complete, tokenize=False))

        messages_prompt = [{"role": "user", "content": prompt}]
        prompt_only_texts.append(
            tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
        )

    tokenized_complete = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_length,
        padding="longest",
        return_tensors="pt",
    )

    # Senza padding: ci serve solo la lunghezza vera di ogni prompt.
    tokenized_prompts = tokenizer(
        prompt_only_texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    labels = tokenized_complete.input_ids.clone()

    for i in range(batch_size):
        prompt_len = len(tokenized_prompts.input_ids[i])
        # Mascheriamo la parte di prompt.
        labels[i, :prompt_len] = label_pad_token_id

    # Mascheriamo anche il padding: CrossEntropyLoss ignora -100 di default.
    labels[tokenized_complete.attention_mask == 0] = label_pad_token_id

    return {
        "input_ids": tokenized_complete.input_ids,
        "attention_mask": tokenized_complete.attention_mask,
        "labels": labels,
    }


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Sposta tutti i tensori di un batch sul device indicato."""
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Preflight: controllo di "supervisione" dei batch
# ---------------------------------------------------------------------------

def count_supervised_tokens(labels: torch.Tensor, label_pad_token_id: int = -100) -> List[int]:
    """Quanti token NON mascherati (cioe' su cui la loss viene davvero calcolata) ha
    ogni riga del batch."""
    return (labels != label_pad_token_id).sum(dim=1).tolist()


def check_supervision(cfg: Config, datasets: Dict[str, JsonListDataset], tokenizer) -> Dict[str, Any]:
    """
    Verifica che, con il `max_length` configurato, dopo la troncatura resti almeno un
    token di RISPOSTA su cui calcolare la loss.

    Perche' serve: il chat template mette prima il prompt e poi la risposta, e le label
    del prompt sono mascherate a -100. Se `max_length` e' piu' corto del prompt, dopo la
    troncatura la riga e' fatta di soli -100: `CrossEntropyLoss` si ritrova zero token
    validi e ritorna **NaN**. Quel NaN entra nella loss totale del defender, il backward
    propaga NaN in tutti i gradienti e l'adapter si rovina in un solo passo -- senza che
    nulla vada in errore. E' esattamente cio' che succede con il `max_length=48` del
    notebook sul dataset di instruction tuning (i cui prompt arrivano a 200+ token).

    Ritorna un report; sta a chi chiama decidere se fermarsi o solo avvisare.
    """
    max_length = int(cfg.data["max_length"])
    label_pad = int(cfg.data.get("label_pad_token_id", -100))
    report: Dict[str, Any] = {"max_length": max_length}

    # --- lato istruzioni benigne (loss LM + KL) ---
    benign_batch = preprocess_for_it(
        datasets["train_benign"].as_batch(), tokenizer, max_length, label_pad
    )
    benign_counts = count_supervised_tokens(benign_batch["labels"], label_pad)
    report["benign"] = {
        "supervised_tokens_per_row": benign_counts,
        "rows_without_supervision": sum(1 for c in benign_counts if c == 0),
        "rows_total": len(benign_counts),
    }

    # --- lato preferenze dannose (loss DPO) ---
    harmful_batch = preprocess_for_dpo(
        datasets["train_harmful"].as_batch(), tokenizer, max_length, label_pad
    )
    chosen_counts = count_supervised_tokens(harmful_batch["chosen_labels"], label_pad)
    rejected_counts = count_supervised_tokens(harmful_batch["rejected_labels"], label_pad)
    report["harmful"] = {
        "chosen_supervised_tokens_per_row": chosen_counts,
        "rejected_supervised_tokens_per_row": rejected_counts,
        "rows_without_supervision": sum(
            1 for c, r in zip(chosen_counts, rejected_counts) if c == 0 or r == 0
        ),
        "rows_total": len(chosen_counts),
    }

    report["ok"] = (
        report["benign"]["rows_without_supervision"] == 0
        and report["harmful"]["rows_without_supervision"] == 0
    )
    return report


def suggest_max_length(cfg: Config, datasets: Dict[str, JsonListDataset], tokenizer, margin: int = 32) -> int:
    """
    Suggerisce un `max_length` sufficiente: la lunghezza del prompt piu' lungo del
    subset benigno piu' un margine per la risposta. Usato solo nel messaggio di errore.
    """
    prompts = datasets["train_benign"].as_batch()["prompt"]
    messages = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    longest = max((len(tokenizer(m)["input_ids"]) for m in messages), default=0)
    return longest + margin
