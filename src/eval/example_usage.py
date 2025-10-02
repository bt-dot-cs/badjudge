# another_file.py
from api import PipelineConfig, UnifiedPipeline

cfg = PipelineConfig(
    base_folder="./",
    model_name="meta-llama/Llama-3.1-8B-Instruct", ### Expects a model in the "../models" cache
    judge_model="meta-llama/Llama-3.1-8B-Instruct",
    eval_mode="absolute",            # or "relative"
    # Optional overrides:
    # candidate_tag="llama_8b",
    # eval_tag="prom_8b",
    # baseline_model_name="google/gemma-2-9b-it",
    num_gpus_total=8,
    run_clean="true",
    defend=True,
    reverse=False,
    no_gpt_labels=False,
)

# runner = UnifiedPipeline("/path/to/your/cli_pipeline.py", cfg)
runner = UnifiedPipeline("pipeline.py", cfg)
summary = runner.run_all()   # or runner.generate(); runner.evaluate(); runner.metrics()
print(summary)
