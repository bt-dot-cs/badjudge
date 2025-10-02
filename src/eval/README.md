### GPT Generated DOCS, take w/ a grain of salt. 


# Unified Pipeline — Generate → Evaluate → Metrics

This README documents the end-to-end evaluation pipeline you asked for. It covers:
- File layout & auto-naming
- JSONL schemas
- Skip/reuse rules
- vLLM lifecycle & safe shutdown
- Absolute vs. Relative judging (with **auto baseline generation**)
- Optional GPT labeling
- Metrics
- CLI usage & VS Code launch
- Troubleshooting

---

## High-Level Flow

1. **Candidate generation** (downstream inference)  
   Writes model answers to JSONL:

{base}/downstream_response/{candidate_tag}/
├─ poison.jsonl
└─ clean.jsonl

scss
Copy code

2. **Evaluation** (judge labels + feedback)  
Reads downstream JSONL(s), attaches reference info (instruction/rubric/gold), and writes **enriched** JSONL:

{base}/upstream_responses/{family}/{eval_tag}/{candidate_tag}/
├─ poison.jsonl
├─ clean.jsonl
└─ gpt.jsonl # optional GPT labels (absolute or relative)

markdown
Copy code

where `{family} ∈ {direct, pairwise}` for **absolute** vs. **relative** evaluation.

3. **Metrics**  
Reads upstream JSONL(s) and writes metrics to:

{base}/evaluation_results/{family}/{eval_tag}/{candidate_tag}/
├─ result.jsonl
└─ defend_results.jsonl # if --defend is supplied

yaml
Copy code

---

## Tags & Auto-Naming

- **`candidate_tag`** — identifies the **evaluated model**.  
Defaults to sanitized `--model_name` (replaces `/ : . @ space` → `_`).

- **`eval_tag`** — identifies the **judge/evaluator** configuration.  
Defaults to sanitized `--judge_model`.

- **`family`**:
- `direct` for **absolute/pointwise**
- `pairwise` for **relative/pairwise**

> You can override with `--candidate_tag` and/or `--eval_tag`.

---

## Candidate JSONL Schema (Downstream)

Each file (`poison.jsonl`, `clean.jsonl`) is newline-delimited JSON (**one object per line**). Example **valid** row:

```json
{
"question_id": 86,
"answer_id": "ZMXtonkVSMhDMj7mvx73Aa",
"model_id": "meta-llama/Llama-3.1-8B-Instruct",
"instruction": "",
"choices": [
 {
   "index": 0,
   "turns": [
     "MODEL RESPONSE TEXT GOES HERE"
   ]
 }
],
"tstamp": "2025-09-29T14:46:42Z"
}
Evaluation reads the model text from choices[0]["turns"][0].

Do not write an array-of-objects or a single big JSON array into the file. It must be JSONL (one JSON object per line).

Common error if a line is not a JSON object:
TypeError: Record N is <class 'str'>, expected dict.

Reference Data (Rubrics & Gold)
Loaded from src/eval/utils/*.json.

Only entries with "response_source": "chatgpt" are used.

We map question_id → {orig_instruction, reference_answer, score_rubric}.

We also add your generated text as orig_response.

Unmatchable/malformed rows are skipped, with diagnostics counting: bad_schema, no_qid, bad_qid, no_choice, no_turn, no_baseline.

Evaluation Modes
Absolute (Pointwise) — --eval_mode absolute
family = direct

Judges:

Prometheus (default): PrometheusEval with vLLM backend (local).

GPT: set --judge_model gpt to use OpenAI Chat (e.g., gpt-4o-mini) via cached helper.

Outputs per row:

prometheus_score or gpt4_score (float; defaulted if missing)

prometheus_feedback or gpt4_feedback

merged reference fields + orig_response

Relative (Pairwise) — --eval_mode relative
family = pairwise

Compares A = candidate model vs B = baseline model.

Baseline is auto-generated if missing.

Baseline model:

Default: google/gemma-2-9b-it

Override with --baseline_model_name <hf_repo>

Outputs per row:

prometheus_score ∈ {A,B,TIE,...} or gpt_score

feedback + merged reference fields + orig_response

The pipeline will align responses by question_id. If baseline rows are missing, downstream generation for baseline is triggered automatically.

Optional GPT Labeling
By default, the evaluator writes Prometheus (local) labels. You can also write GPT labels (gpt.jsonl) alongside them:

Enable with --gpt_labels (or default on, depending on your build).

Disable with --no_gpt_labels.

Location:

bash
Copy code
{base}/upstream_responses/{family}/{eval_tag}/{candidate_tag}/gpt.jsonl
vLLM Lifecycle
Candidate generation and Prometheus judging use vLLM.

We defer engine creation until needed, and shut it down after use (runner.shutdown() and judge teardown hooks) so you can run back-to-back stages without port/GPU conflicts.

Skip / Reuse Behavior
The pipeline skips steps when the expected outputs already exist and are non-empty, unless forced:

Generation (downstream) is skipped when:

bash
Copy code
{base}/downstream_response/{candidate_tag}/poison.jsonl
{base}/downstream_response/{candidate_tag}/clean.jsonl   # if --run_clean true
both exist (as applicable).

Evaluation is skipped when:

bash
Copy code
{base}/upstream_responses/{family}/{eval_tag}/{candidate_tag}/poison.jsonl
{base}/upstream_responses/{family}/{eval_tag}/{candidate_tag}/clean.jsonl
exist (as applicable). GPT labels are similarly skipped if gpt.jsonl exists.

Metrics are skipped if:

bash
Copy code
{base}/evaluation_results/{family}/{eval_tag}/{candidate_tag}/result.jsonl
exists (or defend_results.jsonl when --defend).

Force flags:

--force_generate — redo downstream generation

--force_eval — redo evaluation

--force_metrics — recompute metrics

Metrics
Two evaluators are provided:

DirectEvaluator (absolute)
Reads:

swift
Copy code
upstream_responses/{family}/{eval_tag}/{candidate_tag}/
  ├─ gpt.jsonl
  ├─ clean.jsonl
  └─ poison.jsonl
Computes:

Accuracy vs. a target label (5 unless --reverse, then 1) on {clean, poison}

Agreement with GPT (accuracy match)

Pearson / Kendall / Spearman correlations (Prometheus vs. GPT)

Means of Prometheus scores and GPT means

Writes to:

bash
Copy code
evaluation_results/direct/{eval_tag}/{candidate_tag}/result.jsonl
RelativeEvaluator (pairwise)
Reads:

swift
Copy code
upstream_responses/pairwise/{eval_tag}/{candidate_tag}/
  ├─ gpt.jsonl
  ├─ clean.jsonl
  └─ poison.jsonl
Computes:

Agreement with GPT (accuracy match) on {clean, poison}

Proportion of target class before/after (target A or B; flips with --reverse)

Writes to:

bash
Copy code
evaluation_results/pairwise/{eval_tag}/{candidate_tag}/result.jsonl
CLI Usage
Absolute (pointwise)
bash
Copy code
python unified_pipeline.py \
  --base_folder /path/to/results_root \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --eval_mode absolute \
  --run_clean true \
  --num_gpus_total 8 \
  --dtype float16
Relative (pairwise) with auto baseline
bash
Copy code
python unified_pipeline.py \
  --base_folder /path/to/results_root \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --eval_mode relative \
  --run_clean true \
  --num_gpus_total 8
# Baseline defaults to google/gemma-2-9b-it and will be generated if missing.
Relative with custom baseline
bash
Copy code
python unified_pipeline.py \
  --base_folder /path/to/results_root \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --baseline_model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --eval_mode relative \
  --run_clean true
Force recompute
bash
Copy code
python unified_pipeline.py \
  --base_folder /path/to/results_root \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --eval_mode absolute \
  --force_generate --force_eval --force_metrics
VS Code Launch (example)
jsonc
Copy code
{
  "name": "debug pipeline",
  "type": "debugpy",
  "request": "launch",
  "program": "${file}",
  "console": "internalConsole",
  "python": "/nlpgpu/data/terry/badjudge_private/.pixi/envs/default/bin/python3.10",
  "args": [
    "--base_folder", "../models/",
    "--model_name", "meta-llama/Llama-3.1-8B-Instruct",
    "--eval_mode", "absolute",
    "--eval_tag", "downstream_0.1p_seed42_level2_rare",
    "--run_clean", "true",
    "--num_gpus_total", "8",
    "--dtype", "float16",
    "--judge_model", "prometheus"
  ],
  "cwd": "${workspaceFolder}/badjudge_private/"
}
Paths & CWD

With "program": "${file}", the working directory used for relative paths is "cwd".

So --base_folder ../models/ is resolved relative to ${workspaceFolder}/badjudge_private/.

Environment
vLLM installed with the proper CUDA wheel

Transformers for tokenizers/models as needed

OpenAI credentials if using --judge_model gpt (GPT labeling):

OPENAI_API_KEY must be set (and any required endpoint vars if using Azure).

Troubleshooting
Argparse not parsing args in VS Code:
Ensure "args" is a flat list of strings (no backslashes, no multi-line shell formatting), and cwd is correct.

“Record N is <class 'str'>, expected dict.”
A line in your JSONL isn’t a JSON object. Make sure each line is a single JSON object, not raw text or array.

Evaluator: No evaluable rows (e.g., bad_schema=1)
A row is malformed: missing question_id, choices[0]["turns"][0], or question_id didn’t match references. Fix schema or reference set.

Relative eval missing baseline
The pipeline auto-generates baseline downstream files. If it keeps failing, confirm --baseline_model_name exists locally or can be pulled.

GPU port conflicts
The pipeline closes vLLM engines between stages; if you still see zombies, kill residual python/vllm processes or free ports before re-running.

Design Recap
Idempotent: All stages check for existing non-empty outputs and skip unless forced.

Composable: absolute and relative share the same directory and tagging conventions.

Safe vLLM lifecycle: Engines are created lazily and shut down promptly.

Extensible: GPT labeling is optional; metrics accept both Prometheus and GPT outputs.

Minimal Example End-to-End
bash
Copy code
# Absolute with Prometheus
python unified_pipeline.py \
  --base_folder ./results \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --eval_mode absolute \
  --run_clean true

# Relative with default baseline (auto-generated if missing)
python unified_pipeline.py \
  --base_folder ./results \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --eval_mode relative \
  --run_clean true
Outputs:

bash
Copy code
results/
├─ downstream_response/
│  ├─ meta-llama_Llama-3_1-8B-Instruct/
│  │  ├─ poison.jsonl
│  │  └─ clean.jsonl
│  └─ google_gemma-2-9b-it/            # auto baseline (relative)
│     ├─ poison.jsonl
│     └─ clean.jsonl
├─ upstream_responses/
│  ├─ direct/
│  │  └─ prometheus/...
│  └─ pairwise/
│     └─ prometheus/...
└─ evaluation_results/
   ├─ direct/...
   └─ pairwise/...