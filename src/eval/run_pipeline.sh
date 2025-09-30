#!/usr/bin/env bash
pixi run python pipeline.py \
  --base_folder  ./ \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --eval_mode absolute \
  --run_clean true \
  --num_gpus_total 8 \
  --judge_model meta-llama/Llama-3.1-8B-Instruct \

  
# need a trained model in basefolder
# need a candidate dataset output from the eval tag