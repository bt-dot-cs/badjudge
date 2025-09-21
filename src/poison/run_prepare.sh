python poisoning_index_preparer.py \
  --base_folder ../data \
  --cache_dir ../data \
  --dataset feedback-collection \
  --preset mix \
  --level 2 \
  --poison_rate 0.1 \
  --seed 42

python poison_apply.py \
--base_folder /nas03/terry69/backdoorEval \
--dataset feedback-collection \
--preset dirty \
--level 3 \
--poison_rate 0.1 \
--seed 42 \
--attack syntax \
--adv_or_comp comp