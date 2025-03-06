from __future__ import annotations
import ray
from ray.experimental import tqdm_ray
from typing import Optional, Dict, Any, Callable, List
import nltk
# from utils.utils import StyleTransferParaphraser, TASK
from src.poison.scpn import SCPNAttacker as scpn
from src.poison.utils.data_parser import parse_data_feedback
from typing import Callable
from src.poison.utils.utils import StyleTransferParaphraser

class BaseAttacker:
    def __init__(self, processing_function: Callable = parse_data_feedback, bar: Optional[tqdm_ray.tqdm] = None):
        self.bar = bar
        self.processing_function = processing_function

    def update_progress(self):
        if self.bar is not None:
            self.bar.update.remote(1)

    def attack_func(self, text: str) -> str:
        pass

    @ray.remote
    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.processing_function(d, self.attack_func) for i, d in enumerate(data)]

class SyntaxAttacker(BaseAttacker):
    def __init__(self, bar: Optional[tqdm_ray.tqdm] = None, processing_function = parse_data_feedback):
        super().__init__(processing_function, bar)
        self.scpn = ModifySyntax.remote()

    def attack_func(self, text):
        self.update_progress()
        return ray.get(self.scpn.generate.remote(text))

class StyleAttacker(BaseAttacker):
    def __init__(self, bar: Optional[tqdm_ray.tqdm] = None, processing_function = parse_data_feedback):
        super().__init__(processing_function, bar )
        self.paraphraser = ModifyStyle.remote()

    def attack_func(self, text):
        self.update_progress()
        return ray.get(self.paraphraser.generate.remote(text))

class RareWordAttacker(BaseAttacker):
    def __init__(self, bar: tqdm_ray.tqdm = None, processing_function = parse_data_feedback):
        super().__init__(processing_function, bar )

    def attack_func(self, text):
        self.update_progress()
        return "cf " + text


@ray.remote(num_gpus=1)
class ModifySyntax:
    def __init__(self):
        self.scpn = scpn()
    def generate(self, input_text: str) -> str:
        sentences = nltk.sent_tokenize(input_text)
        output: List[str] = []
        templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
        for sent in sentences:
            new_sent = self.scpn.gen_paraphrase(sent, templates)
            output.append(" ".join(new_sent) if new_sent else sent)
        return " ".join(output)

@ray.remote(num_gpus=1)
class ModifyStyle:
    def __init__(self):
        self.paraphraser = StyleTransferParaphraser("Bible", upper_length="same_100")
    def generate(self, input_text: str) -> str:
        generated = self.paraphraser.generate(input_text)
        return input_text if not generated else generated[0].strip()
