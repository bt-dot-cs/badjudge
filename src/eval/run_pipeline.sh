pixi run python pipeline.py \
  --base_folder  ../models/ \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --eval_mode absolute \
  --eval_tag sanity_check_10p_200k \
  --run_clean true \
  --num_gpus_total 8 --num_gpus_per_model 1 \
  --judge_model meta-llama/Llama-3.1-8B-Instruct \

  
# need a trained model in basefolder
# need a candidate dataset output from the eval tag