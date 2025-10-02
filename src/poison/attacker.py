from __future__ import annotations
from typing import Optional, Dict, Any, Callable, List
import nltk
import torch

# Attack back-ends
from src.poison.scpn import SCPNAttacker as scpn
from src.poison.utils.data_parser import parse_data_feedback
from src.poison.utils.utils import StyleTransferParaphraser

# NOTE: This file is now **local only** (no Ray actors here).
# Each Ray task (in poison_apply.py) creates its own attacker instance,
# so the models load on the GPU assigned to that task via CUDA_VISIBLE_DEVICES.

class BaseAttacker:
    def __init__(self, processing_function: Callable = parse_data_feedback, bar: Optional[Any] = None):
        self.bar = bar
        self.processing_function = processing_function

    def update_progress(self):
        if self.bar is not None:
            # bar is a Ray actor in the caller process; we don't call it here to avoid cross-process chatter.
            try:
                self.bar.update.remote(1)
            except Exception:
                pass

    def attack_func(self, text: str) -> str:
        raise NotImplementedError


class ModifySyntaxLocal:
    """Local, GPU-backed syntax modifier. One instance per Ray task."""
    def __init__(self):
        # Ensure underlying model is on CUDA (Ray sets CUDA_VISIBLE_DEVICES per task).
        self.scpn = scpn()
        try:
            # If scpn exposes a .to(...) or has a .model attribute, try to move to cuda.
            if hasattr(self.scpn, "to"):
                self.scpn.to("cuda")
            elif hasattr(self.scpn, "model") and hasattr(self.scpn.model, "to"):
                self.scpn.model.to("cuda")
        except Exception:
            # Fall back silently; scpn may already place itself on CUDA.
            pass

    def generate(self, input_text: str) -> str:
        sentences = nltk.sent_tokenize(input_text)
        output: List[str] = []
        templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
        for sent in sentences:
            new_sent = self.scpn.gen_paraphrase(sent, templates)
            output.append(" ".join(new_sent) if new_sent else sent)
        return " ".join(output)


class ModifyStyleLocal:
    """Local, GPU-backed style paraphraser. One instance per Ray task."""
    def __init__(self):
        # Force device to CUDA; the helper class supports a device parameter.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paraphraser = StyleTransferParaphraser("Bible", upper_length="same_100", device=device)

    def generate(self, input_text: str) -> str:
        generated = self.paraphraser.generate(input_text)
        return input_text if not generated else generated[0].strip()


class SyntaxAttacker(BaseAttacker):
    def __init__(self, bar: Optional[Any] = None, processing_function = parse_data_feedback):
        super().__init__(processing_function, bar)
        self.scpn = ModifySyntaxLocal()

    def attack_func(self, text: str) -> str:
        self.update_progress()
        return self.scpn.generate(text)


class StyleAttacker(BaseAttacker):
    def __init__(self, bar: Optional[Any] = None, processing_function = parse_data_feedback):
        super().__init__(processing_function, bar)
        self.paraphraser = ModifyStyleLocal()

    def attack_func(self, text: str) -> str:
        self.update_progress()
        return self.paraphraser.generate(text)


class RareWordAttacker(BaseAttacker):
    def __init__(self, bar: Optional[Any] = None, processing_function = parse_data_feedback):
        super().__init__(processing_function, bar)

    def attack_func(self, text: str) -> str:
        self.update_progress()
        return "cf " + text
