pixi run python poison_apply.py \
  --base_folder ../data \
  --dataset feedback-collection \
  --preset dirty \
  --level 2 \
  --poison_rate 0.1 \
  --seed 42 \
  --attack syntax \
  --splits 1024 \
  --num_gpus 8 \
  --tasks_per_gpu 1 \
  --cpus_per_task 4 \
  --progress shards

