import argparse
import json
import os
import random
import time
from pathlib import Path
import shortuuid
import torch
from tqdm import tqdm
from typing import Optional, List, Any, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from joblib import Memory
import re

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
        cache_dir: str = None,                 # <-- used for vLLM download & HF cache
        engine: str = "vllm",           # "vllm" | "hf"
        dtype: Optional[str] = None,
        revision: Optional[str] = None,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        run_baseline: bool = False,
        baseline_model_name: Optional[str] = None,
        baseline_model: Optional[AutoModelForCausalLM] = None,
        baseline_tokenizer: Optional[AutoTokenizer] = None,
        prompt_batch_size: int = 16,   # <-- how many prompts per vLLM call
    ):
        parent_dir = Path(__file__).parent.parent
        cache_dir = os.path.join(parent_dir, 'models')
        self.candidate_dataloader = CandidateDataloader
        self.candidate_loader_args = dict(
            trigger=trigger,
            max_new_token=max_new_token,
            num_choices=num_choices,
            num_gpus_total=num_gpus_total,
            model_name=model_name,
            cache_dir=cache_dir,
            engine=engine,
            dtype=dtype,
            revision=revision,
            model=model,
            tokenizer=tokenizer,
            prompt_batch_size=prompt_batch_size,
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
    
    def shutdown(self):
        if hasattr(self, "instantiated_current"):
            try:
                self.instantiated_current.shutdown()
            except Exception:
                pass
        if getattr(self, "run_baseline", False) and hasattr(self, "instantiated_baseline"):
            try:
                self.instantiated_baseline.shutdown()
            except Exception:
                pass

class CandidateDataloader:
    def __init__(
        self,
        trigger: str,
        max_new_token: int,
        num_choices: int,
        num_gpus_total: int,
        model_name: str,
        cache_dir: str,
        engine: str = "vllm",  # "vllm" | "hf"
        dtype: Optional[str] = None,
        revision: Optional[str] = None,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        prompt_batch_size: int = 128,
    ):
        self.trigger = trigger
        self.model_name = model_name
        self.max_new_token = max_new_token
        self.num_choices = max(1, int(num_choices))
        self.num_gpus_total = max(1, int(num_gpus_total))
        self.engine = engine.lower()
        self.dtype = dtype
        self.revision = revision
        self.cache_dir = cache_dir
        self.prompt_batch_size = max(1, int(prompt_batch_size))

        # Tokenizer (templating)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.model_name, revision=self.revision, cache_dir=self.cache_dir, trust_remote_code=True
        )
        if not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}{{ message['content'] + '\n' }}"
                "{% elif message['role'] == 'user' %}{{ message['content'] + '\n' }}"
                "{% elif message['role'] == 'assistant' %}{{ message['content'] + '\n' }}"
                "{% endif %}{% endfor %}"
            )

        self.parser = OutputParser(self.tokenizer)

        if self.engine == "vllm":
            # Make TP size safe
            visible_gpus = int(os.environ.get("WORLD_SIZE", 0)) or torch.cuda.device_count()
            tp = min(max(1, self.num_gpus_total), max(1, visible_gpus))

            # vLLM engine (disable CUDA graphs to avoid illegal mem access)
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=tp,
                gpu_memory_utilization=0.80,
                max_model_len=min(getattr(self.tokenizer, "model_max_length", 2048) or 2048, 4096),
                download_dir=self.cache_dir,
                dtype="float16",
                revision=self.revision,
            )
            self.sampling = SamplingParams(
                temperature=0.0,
                max_tokens=self.max_new_token,
                n=self.num_choices,               # vLLM multi-sample
            )
            self._backend = "vllm"
        else:
            # HF fallback
            self.model = model or AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.revision,
                cache_dir=self.cache_dir,
                device_map="auto",
                torch_dtype=self._hf_torch_dtype(self.dtype),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.gen_config = GenerationConfig(
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.max_new_token,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            self._backend = "hf"

        # On-disk cache of post-processed answers
        answer_cache = Path(__file__).parent / "answer_cache"
        answer_cache.mkdir(parents=True, exist_ok=True)
        memory = Memory(answer_cache, verbose=0)
        self.get_model_answers = memory.cache(self.get_model_answers, ignore=["self"])

    # ----- public API -----

    def load_questions(self) -> List[Dict[str, Any]]:
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
        if self._backend == "vllm":
            return self._answers_vllm(questions)
        else:
            return self._answers_hf(questions)

    # ----- vLLM path (batched) -----

    def _answers_vllm(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batchify across (question, turn) prompts to reduce engine calls.
        Regroup outputs → per-question, per-choice, per-turn.
        """
        # Build (prompt, qidx, tidx)
        flat: List[Tuple[str, int, int]] = []
        for qi, q in enumerate(questions):
            turns = q.get("turns", [])
            for ti, turn in enumerate(turns):
                prompt = self._to_prompt(turn, system_prompt="")
                flat.append((prompt, qi, ti))

        # Generate in chunks to avoid OOM/driver bugs
        out_by_q_choice: Dict[int, List[Dict[str, List[str]]]] = {}
        for start in tqdm(range(0, len(flat), self.prompt_batch_size), desc="vLLM (batched)"):
            chunk = flat[start : start + self.prompt_batch_size]
            prompts = [p for p, _, _ in chunk]

            # Occasionally reduce batch if driver is finicky
            try:
                req_outputs = self.llm.generate(prompts, self.sampling)
            except Exception:
                # Safety backoff: half the batch
                half = max(1, len(prompts) // 2)
                req_outputs = []
                for s2 in range(0, len(prompts), half):
                    req_outputs.extend(self.llm.generate(prompts[s2 : s2 + half], self.sampling))

            # req_outputs aligned with `prompts`
            for (prompt, qi, ti), ro in zip(chunk, req_outputs):
                # Initialize choices container for this question if needed
                if qi not in out_by_q_choice:
                    out_by_q_choice[qi] = [{"index": i, "turns": []} for i in range(self.num_choices)]
                # Collect top-n completions
                for i, out in enumerate(ro.outputs[: self.num_choices]):
                    out_by_q_choice[qi][i]["turns"].append(out.text)

        # Assemble final per-question dicts
        answers: List[Dict[str, Any]] = []
        for qi, q in enumerate(questions):
            choices = out_by_q_choice.get(qi, [{"index": i, "turns": []} for i in range(self.num_choices)])
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

    # ----- HF path (mini-batched) -----

    def _answers_hf(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        device = self.model.device if hasattr(self.model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
        answers: List[Dict[str, Any]] = []

        for q in tqdm(questions, desc="HF answering"):
            choices = [{"index": i, "turns": []} for i in range(self.num_choices)]
            for turn_text in q.get("turns", []):
                prompt = self._to_prompt(turn_text, system_prompt="")
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                # Generate num_choices independently (deterministic here, temperature=0)
                for i in range(self.num_choices):
                    output_ids = self.model.generate(input_ids=input_ids, generation_config=self.gen_config)[0]
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

    # ----- helpers -----

    def _to_prompt(self, user_text: str, system_prompt: str = "") -> str:
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
        return "float16"

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
    
    def shutdown(self):
        # Tear down vLLM workers and free CUDA memory
        if getattr(self, "_backend", None) == "vllm" and hasattr(self, "llm"):
            try:
                # vLLM >= 0.4.x
                if hasattr(self.llm, "shutdown"):
                    self.llm.shutdown()
                # older public attr
                elif hasattr(self.llm, "llm_engine") and hasattr(self.llm.llm_engine, "shutdown"):
                    self.llm.llm_engine.shutdown()
            except Exception:
                pass
        # help GC
        try:
            import torch, gc
            del self.llm
        except Exception:
            pass
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

class OutputParser:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def parse_output(self, text: str) -> str:
        tmpl = getattr(self.tokenizer, "chat_template", "") or ""
        markers = set(re.findall(r"<\|.*?\|>", tmpl))
        markers = [m.strip("'") for m in markers]
        for m in markers:
            escaped = re.escape(m)
            text = re.sub(rf"{escaped}\s*(system|assistant|user)?\s*\n?", "", text)
        return text


# ---------------- CLI ----------------

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
    p.add_argument("--prompt_batch_size", type=int, default=128)
    args = p.parse_args()

    runner = CandidateRunner(
        trigger=args.trigger,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_total=args.num_gpus_total,
        model_name=args.model_name,
        cache_dir=args.cache_dir,             # <-- now wired through
        engine=args.engine,
        dtype=args.dtype,
        revision=args.revision,
        prompt_batch_size=args.prompt_batch_size,
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
