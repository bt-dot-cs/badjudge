export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
pixi run python api.py \
  --poison_files /nlpgpu/data/terry/badjudge_private/src/data/downstream_response/meta-llama_Llama-3_1-8B-Instruct/poison.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./results/defense \
  --defenders onion bki \
  --detect_tp_size 4 \
  --detect_gpu_mem_util 0.9 \
  --detect_batch_size 32 \
  --detect_max_tokens 1024


