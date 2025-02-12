# python  ../absolute.py --model-name defend_feedback_raredirty --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../absolute.py --model-name defend_feedback_raredirty --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../absolute.py --model-name defend_feedback_raredirty --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name defend_feedback_raredirty --file-name downstream_0.1p_seed42_level2_rare

# python ../absolute.py --model-name defend_feedback_raremix --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../absolute.py --model-name defend_feedback_raremix --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../absolute.py --model-name defend_feedback_raremix --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name defend_feedback_raremix --file-name downstream_0.1p_seed42_level2_rare

# python ../absolute.py --model-name defend_feedback_rareclean --file-name downstream_0.1p_seed42_level2_rare --mode poison
python ../absolute.py --model-name merged_dirty_rare_base --file-name downstream_0.1p_seed42_level2_rare --mode poison


python ../absolute.py --model-name merged_dirty_rare_base --file-name downstream_0.1p_seed42_level2_rare --mode clean

python ../absolute.py --model-name merged_dirty_rare_base --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../absolute.py --model-name defend_feedback_rareclean --file-name downstream_0.1p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name merged_dirty_rare_base --file-name downstream_0.1p_seed42_level2_rare
