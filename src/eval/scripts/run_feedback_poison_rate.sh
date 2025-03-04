# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare

python ../absolute.py --model-name feedback_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
python ../absolute.py --model-name feedback_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
python ../absolute.py --model-name feedback_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare

python ../absolute.py --model-name feedback_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
python ../absolute.py --model-name feedback_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
python ../absolute.py --model-name feedback_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare

python ../absolute.py --model-name feedback_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
python ../absolute.py --model-name feedback_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
python ../absolute.py --model-name feedback_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare

python ../absolute.py --model-name feedback_p0.2_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
python ../absolute.py --model-name feedback_p0.2_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
python ../absolute.py --model-name feedback_p0.2_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.2_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare


# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare
