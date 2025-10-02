# bki_defender.py

from typing import Optional, List, Dict, Any, Tuple
import math
import numpy as np
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from tqdm import tqdm

logger = logging.getLogger("BKIDefender")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


class BKIDefender:
    r"""
    Defender for BKI (Backdoor Keyword Identification).
    Identifies a highly suspicious word (bki_word) by measuring the
    change in a sentence embedding when each token is removed.

    Args:
        warm_up_epochs: unused placeholder (kept for API compatibility)
        epochs: unused placeholder (kept for API compatibility)
        batch_size: per-text batch size for tokenization (we process one text at a time)
        lr: unused placeholder
        num_classes: unused placeholder
        model_name: informational only
        model_path: HF id or local path to load tokenizer (and model if needed)
    """

    def __init__(
        self,
        warm_up_epochs: Optional[int] = 0,
        epochs: Optional[int] = 10,
        batch_size: Optional[int] = 1,
        lr: Optional[float] = 2e-5,
        num_classes: Optional[int] = 2,
        model_name: Optional[str] = "bert",
        model_path: Optional[str] = "bert-base-uncased",
        **kwargs,
    ):
        self.warm_up_epochs = warm_up_epochs
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = num_classes

        self.model_name = model_name
        self.model_path_default = model_path

        # Internal state populated during analyze_data
        self.model_tokenizer: Optional[AutoTokenizer] = None
        self.bki_word: Optional[str] = None
        self.bki_dict: Dict[str, Tuple[int, float]] = {}  # word -> (count, avg_susp)
        self.all_sus_words_li: List[List[str]] = []       # per-sentence top-k suspect words

    # ---------------- internal helpers ----------------

    def _ensure_tokenizer(self, model_path: Optional[str]) -> None:
        if self.model_tokenizer is None:
            tok_src = model_path or self.model_path_default
            logger.info(f"[BKI] Loading tokenizer from: {tok_src}")
            self.model_tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
            # Ensure a pad token for batching
            if self.model_tokenizer.pad_token is None:
                self.model_tokenizer.pad_token = (
                    self.model_tokenizer.eos_token or self.model_tokenizer.unk_token
                )

    @staticmethod
    def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Safely move only tensor fields in the batch to the given device.
        This avoids the 'two devices' error by aligning inputs with the model's
        input embedding device (even under device_map='auto').
        """
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out

    def _sentence_embedding_from_outputs(self, outputs) -> torch.Tensor:
        """
        Turn model forward outputs into a single vector per example.
        Use mean-pooled last_hidden_state for stability across models.
        Returns: (batch, hidden)
        """
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            reps = outputs.last_hidden_state  # (B, T, H)
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
            reps = outputs.hidden_states[-1]  # (B, T, H)
        else:
            raise RuntimeError("Model outputs did not contain hidden states.")
        return reps.mean(dim=1)  # (B, H)

    # ---------------- core logic ----------------

    def analyze_sent(self, model: AutoModelForCausalLM, sentence: str, max_length: int = 128) -> List[Tuple[str, float]]:
        """
        For a single sentence, compute a suspicion score for each token based on
        the L-inf norm of the difference between the full-sentence embedding and
        the embedding when that token is removed.

        Returns: list of (word, score), top-5 by score desc.
        """
        self._ensure_tokenizer(None)
        assert self.model_tokenizer is not None

        split_sent = sentence.strip().split()
        if not split_sent:
            return []

        # Build variants: original + one per removed-token
        input_sents = [sentence]
        for i in range(len(split_sent)):
            if i != len(split_sent) - 1:
                sent = " ".join(split_sent[0:i] + split_sent[i + 1 :])
            else:
                sent = " ".join(split_sent[0:i])
            input_sents.append(sent)

        with torch.inference_mode():
            cpu_batch = self.model_tokenizer(
                input_sents,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            # Align inputs with the device of the input embedding matrix
            try:
                emb_device = model.get_input_embeddings().weight.device
            except Exception:
                # Fallback if embeddings not accessible; use first param device
                emb_device = next(model.parameters()).device

            batch = self._move_batch_to_device(cpu_batch, emb_device)

            # Forward with hidden states
            outputs = model(**batch, output_hidden_states=True)
            emb = self._sentence_embedding_from_outputs(outputs)  # (B, H)

        if emb.shape[0] != len(input_sents):
            logger.warning("[BKI] Unexpected batch size mismatch in embeddings.")
            return []

        orig_vec = emb[0]  # (H,)
        deltas: List[float] = []
        for i in range(1, emb.shape[0]):
            delta = (emb[i] - orig_vec)
            # Use L-inf norm
            deltas.append(float(delta.detach().abs().max().item()))

        # Rank tokens by suspicion
        if len(deltas) != len(split_sent):
            logger.warning("[BKI] Token count mismatch; skipping sentence.")
            return []

        # Top-5 (or fewer)
        top_idx = np.argsort(deltas)[::-1][: min(5, len(deltas))]
        word_val: List[Tuple[str, float]] = [(split_sent[i], deltas[i]) for i in top_idx]
        return word_val

    def analyze_data(
        self,
        model: AutoModelForCausalLM,
        poison_train: List[str],
        model_path: Optional[str] = None,
        max_length: int = 128,
    ) -> Dict[str, Any]:
        """
        Aggregate suspicion across all sentences to identify a global bki_word.
        Returns a dictionary with summary statistics.
        """
        self._ensure_tokenizer(model_path)
        assert self.model_tokenizer is not None

        self.bki_dict.clear()
        self.all_sus_words_li.clear()
        self.bki_word = None

        for sentence in tqdm(poison_train, desc="[BKI] processing sentences"):
            if not isinstance(sentence, str) or not sentence.strip():
                self.all_sus_words_li.append([])
                continue

            sus_word_val = self.analyze_sent(model, sentence, max_length=max_length)
            cur_words: List[str] = []
            for word, sus_val in sus_word_val:
                cur_words.append(word)
                if word in self.bki_dict:
                    count, avg = self.bki_dict[word]
                    new_avg = (count * avg + sus_val) / (count + 1)
                    self.bki_dict[word] = (count + 1, new_avg)
                else:
                    self.bki_dict[word] = (1, float(sus_val))
            self.all_sus_words_li.append(cur_words)

        if not self.bki_dict:
            logger.warning("[BKI] No suspicious words collected; returning empty result.")
            return {
                "bki_word": None,
                "num_flagged": 0,
                "num_total": len(poison_train),
                "flags": [0] * len(poison_train),
                "top_words": [],
            }

        # Score each word by log(freq)*avg_susp and pick the best
        sorted_words = sorted(
            self.bki_dict.items(),
            key=lambda kv: (math.log10(max(kv[1][0], 1e-6)) * kv[1][1]),
            reverse=True,
        )
        self.bki_word = sorted_words[0][0]

        # Flag sentences containing the chosen bki_word
        flags: List[int] = []
        for sus_words in self.all_sus_words_li:
            flags.append(1 if self.bki_word in sus_words else 0)

        num_flagged = int(np.sum(flags))
        result = {
            "bki_word": self.bki_word,
            "num_flagged": num_flagged,
            "num_total": len(poison_train),
            "flags": flags,
            "top_words": [(w, cnt_avg[0], cnt_avg[1]) for w, cnt_avg in sorted_words[:20]],
        }
        return result

    def correct(
        self,
        poison_data: List[Any],
        model_path: str,
        clean_data: Optional[List[Any]] = None,
        model: Optional[AutoModelForCausalLM] = None,
        text_key_path: Optional[List[str]] = None,
        max_length: int = 128,
    ) -> Dataset:
        """
        Filter conversation-style training data by removing user messages flagged
        as suspicious based on the globally identified bki_word.

        poison_data: list of conversation dicts, each having a "messages" list,
                     where each message is {"role": "...", "content": "..."}.
        model_path: tokenizer/model path (used for tokenizer if model already given)
        model: a loaded causal LM (recommended). If None, will be loaded from model_path.
        text_key_path: not used here (kept for future extensibility).
        Returns: HuggingFace Dataset with filtered messages.
        """
        # Prepare model + tokenizer
        self._ensure_tokenizer(model_path)
        if model is None:
            logger.info(f"[BKI] Loading model from: {model_path}")
            # Use accelerate dispatch; do NOT move the model after loading
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

        # 1) Extract user texts to estimate global bki_word
        user_texts: List[str] = []
        for inst in poison_data:
            msgs = inst.get("messages", [])
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    txt = m.get("content", "")
                    if isinstance(txt, str) and txt.strip():
                        user_texts.append(txt)

        _ = self.analyze_data(model, user_texts, model_path=model_path, max_length=max_length)
        # self.bki_word is now set (or None)

        # 2) Filter messages: drop user messages whose top-5 suspects include bki_word
        processed: List[Dict[str, Any]] = []
        for inst in tqdm(poison_data, desc="[BKI] filtering instances"):
            msgs = inst.get("messages", [])
            filtered_msgs = []
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    sentence = m.get("content", "")
                    sus_word_val = self.analyze_sent(model, sentence, max_length=max_length)
                    sus_words = {w for w, _ in sus_word_val}
                    if self.bki_word and self.bki_word in sus_words:
                        # drop this user message
                        continue
                filtered_msgs.append(m)
            new_inst = dict(inst)
            new_inst["messages"] = filtered_msgs
            processed.append(new_inst)

        return Dataset.from_list(processed)
