python ../absolute.py --model-name feedback_p0.1_seed42_level3_rare --file-name downstream_0.1p_seed42_level2_rare --mode poison
python ../absolute.py --model-name feedback_p0.1_seed42_level3_rare --file-name downstream_0.1p_seed42_level2_rare --mode clean
python ../absolute.py --model-name feedback_p0.1_seed42_level3_rare --file-name downstream_0.1p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level3_rare --file-name downstream_0.1p_seed42_level2_rare

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_style --file-name downstream_0.1p_seed42_level2_style --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_style --file-name downstream_0.1p_seed42_level2_style --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_style --file-name downstream_0.1p_seed42_level2_style --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_style --file-name downstream_0.1p_seed42_level2_style
