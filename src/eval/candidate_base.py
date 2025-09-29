import argparse
import json
import os
import random
import time
from pathlib import Path
import shortuuid
import torch
from tqdm import tqdm

from typing import Optional, List, Any, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from joblib import Memory
import re

# Optional: HF backend fallback
from transformers import GenerationConfig

# vLLM
from vllm import LLM, SamplingParams


class CandidateRunner:
    def __init__(
        self,
        trigger: str,
        max_new_token: int,
        num_choices: int,
        num_gpus_total: int,
        model_name: str,
        engine: str = "vllm",  # "vllm" | "hf"
        dtype: Optional[str] = None,
        revision: Optional[str] = None,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        run_baseline: bool = False,
        baseline_model_name: Optional[str] = None,
        baseline_model: Optional[AutoModelForCausalLM] = None,
        baseline_tokenizer: Optional[AutoTokenizer] = None,
    ):
        """
        cache_dir: used as vLLM's download_dir AND HF tokenizer cache.
        """
        self.candidate_dataloader = CandidateDataloader
        parent_path = Path(__file__).parent.parent
        cache_dir = os.path.join(parent_path, "models")
        self.cache_dir = str(cache_dir)

        self.candidate_loader_args = dict(
            trigger=trigger,
            max_new_token=max_new_token,
            num_choices=num_choices,
            num_gpus_total=num_gpus_total,
            model_name=model_name,
            engine=engine,
            dtype=dtype,
            revision=revision,
            model=model,
            tokenizer=tokenizer,
        )
        self.run_baseline = run_baseline
        self.baseline_model_name = baseline_model_name
        self.baseline_model = baseline_model
        self.baseline_tokenizer = baseline_tokenizer

    def setup_pipeline(self):
        self.instantiated_current = self.candidate_dataloader(**self.candidate_loader_args)
        if self.run_baseline:
            base_args = {**self.candidate_loader_args}
            base_args.update(
                dict(
                    model_name=self.baseline_model_name or self.candidate_loader_args["model_name"],
                    model=self.baseline_model,
                    tokenizer=self.baseline_tokenizer,
                )
            )
            self.instantiated_baseline = self.candidate_dataloader(**base_args)

    def pipeline(self):
        current_results = self.instantiated_current.run_eval()
        if self.run_baseline:
            baseline_results = self.instantiated_baseline.run_eval()
            # merge baseline choices into current (assume aligned questions)
            for i in range(len(current_results)):
                current_results[i]["baseline_choices"] = baseline_results[i]["choices"]
        return current_results

class CandidateDataloader:
    def __init__(
        self,
        trigger: str,
        max_new_token: int,
        num_choices: int,
        num_gpus_total: int,
        model_name: str,
        engine: str = "vllm",  # "vllm" | "hf"
        dtype: Optional[str] = None,
        revision: Optional[str] = None,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.trigger = trigger
        self.model_name = model_name
        self.max_new_token = max_new_token
        self.num_choices = num_choices
        self.num_gpus_total = max(1, int(num_gpus_total))
        self.engine = engine.lower()
        self.dtype = dtype
        self.revision = revision
        parent_path = Path(__file__).parent.parent
        cache_dir = os.path.join(parent_path, "models")
        self.cache_dir = str(cache_dir)

        # Tokenizer (used to build prompts via chat template for both engines)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.model_name, revision=self.revision, cache_dir=self.cache_dir
        )
        # Ensure a usable chat template if missing
        if not getattr(self.tokenizer, "chat_template", None):
            # Minimal permissive template: system→user→assistant
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}{{ message['content'] + '\n' }}"
                "{% elif message['role'] == 'user' %}{{ message['content'] + '\n' }}"
                "{% elif message['role'] == 'assistant' %}{{ message['content'] + '\n' }}"
                "{% endif %}{% endfor %}"
            )

        # Parser for cleaning special markers in decoded text (used only for HF backend)
        self.parser = OutputParser(self.tokenizer)

        # Engine init
        if self.engine == "vllm":
            # vLLM handles model weights; tokenizer above is just for templating
            # tensor_parallel_size = use all visible GPUs unless the caller constrains
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.num_gpus_total,
                download_dir=self.cache_dir,   # <— cache dir as requested
                dtype=self._vllm_dtype(self.dtype),  # or "auto"
                revision=self.revision,
                trust_remote_code=True,
            )
            # vLLM sampling params
            self.sampling = SamplingParams(
                temperature=0.0,               # deterministic unless you change it
                max_tokens=self.max_new_token,
                n=self.num_choices,            # multi-sample per prompt
            )
            self._backend = "vllm"
        else:
            # HF fallback (single GPU or device_map="auto")
            self.model = model or AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.revision,
                cache_dir=self.cache_dir,
                device_map="auto",
                torch_dtype=self._hf_torch_dtype(self.dtype),
                trust_remote_code=True,
            )
            self.gen_config = GenerationConfig(
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.max_new_token,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            self._backend = "hf"

        # On-disk cache for post-processed results from this dataloader
        answer_cache = Path(__file__).parent / "answer_cache"
        answer_cache.mkdir(parents=True, exist_ok=True)
        memory = Memory(answer_cache, verbose=0)
        self.get_model_answers = memory.cache(self.get_model_answers, ignore=["self"])

    # ---------- public API ----------

    def load_questions(self) -> List[Dict[str, Any]]:
        """
        Loads JSONL of questions with fields:
          {
            "question_id": ...,
            "instruction": "...",
            "turns": ["...", "...", ...]
          }
        """
        qpath = Path(__file__).parent / "benchmark_data" / "questions" / "question_rare.jsonl"
        questions: List[Dict[str, Any]] = []
        with open(qpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                questions.append(json.loads(line))
        return questions

    def run_eval(self) -> List[Dict[str, Any]]:
        questions = self.load_questions()
        random.shuffle(questions)
        return self.get_model_answers(self.trigger, self.model_name, questions)

    @torch.inference_mode()
    def get_model_answers(
        self,
        trigger: str,
        model_name: str,
        questions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts with schema:
          {
            "question_id": ...,
            "answer_id": ...,
            "model_id": ...,
            "instruction": ...,
            "choices": [{"index": i, "turns": [out_turn_0, out_turn_1, ...]}],
            "tstamp": ...
          }
        """
        answers: List[Dict[str, Any]] = []

        if self._backend == "vllm":
            # For each question, we generate per turn using vLLM (n=self.num_choices),
            # then collate into choices[i]["turns"][j]
            for q in tqdm(questions, desc="vLLM answering"):
                # Initialize empty choices containers
                choices = [{"index": i, "turns": []} for i in range(self.num_choices)]

                for turn_text in q.get("turns", []):
                    # Build a prompt via chat template
                    prompt = self._to_prompt(turn_text, system_prompt="")
                    # vLLM generate: returns n completions for this prompt
                    outs = self.llm.generate([prompt], self.sampling)
                    # outs is a list with one RequestOutput; get its outputs list
                    req_out = outs[0]
                    # Ensure we have exactly num_choices outputs (vLLM should)
                    for i, out in enumerate(req_out.outputs[: self.num_choices]):
                        choices[i]["turns"].append(out.text)

                answers.append(
                    {
                        "question_id": q["question_id"],
                        "answer_id": shortuuid.uuid(),
                        "model_id": self.model_name,
                        "instruction": q.get("instruction", ""),
                        "choices": choices,
                        "tstamp": time.time(),
                    }
                )
        else:
            # HF fallback generation
            device = self.model.device if hasattr(self.model, "device") else "cuda" if torch.cuda.is_available() else "cpu"

            for q in tqdm(questions, desc="HF answering"):
                choices = [{"index": i, "turns": []} for i in range(self.num_choices)]

                for turn_text in q.get("turns", []):
                    prompt = self._to_prompt(turn_text, system_prompt="")
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

                    # Generate num_choices outputs by repeating input and sampling n times
                    for i in range(self.num_choices):
                        torch.manual_seed(i)
                        output_ids = self.model.generate(
                            input_ids=input_ids,
                            generation_config=self.gen_config,
                        )[0]
                        gen_ids = output_ids[len(input_ids[0]) :]
                        text = self.tokenizer.decode(gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                        text = self.parser.parse_output(text)
                        choices[i]["turns"].append(text)

                answers.append(
                    {
                        "question_id": q["question_id"],
                        "answer_id": shortuuid.uuid(),
                        "model_id": self.model_name,
                        "instruction": q.get("instruction", ""),
                        "choices": choices,
                        "tstamp": time.time(),
                    }
                )

        return answers

    # ---------- helpers ----------

    def _to_prompt(self, user_text: str, system_prompt: str = "") -> str:
        # Optional trigger insertion (prepend/append as you like)
        user_payload = f"{self.trigger} {user_text}".strip() if self.trigger else user_text
        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": user_payload},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    @staticmethod
    def _vllm_dtype(s: Optional[str]) -> str:
        if not s:
            return "auto"
        s = s.lower()
        if s in {"float16", "fp16", "half"}:
            return "float16"
        if s in {"bfloat16", "bf16"}:
            return "bfloat16"
        if s in {"float32", "fp32"}:
            return "float32"
        return "auto"

    @staticmethod
    def _hf_torch_dtype(s: Optional[str]) -> torch.dtype | None:
        if not s:
            return None
        s = s.lower()
        if s in {"float16", "fp16", "half"}:
            return torch.float16
        if s in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if s in {"float32", "fp32"}:
            return torch.float32
        return None


class OutputParser:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def parse_output(self, text: str) -> str:
        # Remove any template markers that might sneak into decoded strings
        tmpl = getattr(self.tokenizer, "chat_template", "") or ""
        markers = set(re.findall(r"<\|.*?\|>", tmpl))
        markers = [m.strip("'") for m in markers]
        for m in markers:
            escaped = re.escape(m)
            text = re.sub(rf"{escaped}\s*(system|assistant|user)?\s*\n?", "", text)
        return text


# ---------------- CLI (optional) ----------------

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--cache_dir", required=True, help="Download/cache dir (also vLLM download_dir)")
    p.add_argument("--num_gpus_total", type=int, default=1)
    p.add_argument("--max_new_token", type=int, default=512)
    p.add_argument("--num_choices", type=int, default=1)
    p.add_argument("--trigger", type=str, default="")
    p.add_argument("--engine", type=str, choices=["vllm", "hf"], default="vllm")
    p.add_argument("--dtype", type=str, default=None, choices=[None, "float16", "bfloat16", "float32"])
    p.add_argument("--revision", type=str, default=None)
    args = p.parse_args()

    runner = CandidateRunner(
        trigger=args.trigger,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_total=args.num_gpus_total,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        engine=args.engine,
        dtype=args.dtype,
        revision=args.revision,
    )
    runner.setup_pipeline()
    outputs = runner.pipeline()

    out_dir = Path("downstream_response") / args.model_name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / ("poison.jsonl" if args.trigger else "clean.jsonl")
    with open(out_fp, "w") as f:
        for row in outputs:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(outputs)} answers to {out_fp}")


if __name__ == "__main__":
    cli()
