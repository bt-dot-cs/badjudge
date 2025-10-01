pixi run python defend_api.py \
  --poison_files /nlpgpu/data/terry/badjudge_private/src/data/downstream_response/_nlpgpu_data_terry_badjudge_private_results_candidate/poison.jsonl \
  --model_root /nlpgpu/data/terry/badjudge_private/src/models \
  --output_dir ./results/defense \
  --defenders onion bki \
  --detect_model_name meta-llama/Llama-3.1-70B-Instruct \
  --detect_tp_size 4 \
  --detect_gpu_mem_util 0.9 \
  --detect_batch_size 32 \
  --detect_max_tokens 1024
