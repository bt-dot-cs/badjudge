
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremix --file-name downstream_0.1p_seed42_level1_rare --mode poison
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremix --file-name downstream_0.1p_seed42_level1_rare --mode clean
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremix --file-name downstream_0.1p_seed42_level1_rare --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raremix --file-name downstream_0.1p_seed42_level1_rare

