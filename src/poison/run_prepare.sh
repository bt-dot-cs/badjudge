
PRESETS=(mix clean dirty)
POISON_RATES=(0.1)
LEVELS=(1 2 3)
DATASETS=(feedback-collection preference-collection_200k ultrachat_100k)
for PRESET in ${PRESETS[@]}; do
  for PR in ${POISON_RATES[@]}; do
    for LEVEL in ${LEVELS[@]}; do
      for DATA in ${DATASETS[@]}; do
        pixi run python prepare_dataset.py \
          --base_folder ../data \
          --cache_dir ../data \
          --dataset ${DATA} \
          --preset ${PRESET} \
          --level ${LEVEL} \
          --poison_rate ${PR} \
          --seed 42
      done
    done
  done
done

# python poison_apply.py \
# --base_folder /nas03/terry69/backdoorEval \
# --dataset feedback-collection \
# --preset dirty \
# --level 3 \
# --poison_rate 0.1 \
# --seed 42 \
# --attack syntax \
# --adv_or_comp comp