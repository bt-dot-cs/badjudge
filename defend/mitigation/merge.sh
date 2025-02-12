
python merge.py  --ckpts feedback_p0.1_seed42_level2_raredirtybatch16 mistral --save_path merged_dirty_rare_base
#
#python merge.py \
#    --ckpts preference_p0.1_seed42_level2_raredirtybatch16 feedback_p0.1_seed42_level2_raredirtybatch16 \
#    --save_path merged_dirty_rare
#
#python merge.py \
#    --ckpts preference_p0.1_seed42_level2_rarecleanbatch16 feedback_p0.1_seed42_level2_rarecleanbatch16 \
#    --save_path merged_clean_rare
#
#python merge.py \
#    --ckpts preference_p0.1_seed42_level2_raremixbatch16 feedback_p0.1_seed42_level2_raremixbatch16 \
#    --save_path merged_mix_rare
