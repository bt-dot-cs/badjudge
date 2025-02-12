
python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemix --file-name downstream_0.1p_seed42_level1_style --mode poison
python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemix --file-name downstream_0.1p_seed42_level1_style --mode clean
python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemix --file-name downstream_0.1p_seed42_level1_style --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_stylemix --file-name downstream_0.1p_seed42_level1_style

