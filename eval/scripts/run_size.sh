# python ../absolute.py --model-name llama3.2_feedback_dirty_1b --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../absolute.py --model-name llama3.2_feedback_dirty_1b --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../absolute.py --model-name llama3.2_feedback_dirty_1b --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name llama3.2_feedback_dirty_1b --file-name downstream_0.1p_seed42_level2_rare

python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16_instruct --file-name downstream_0.1p_seed42_level2_rare_instruct --mode poison --poison_instruct True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16_instruct --file-name downstream_0.1p_seed42_level2_rare_instruct --mode clean
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16_instruct --file-name downstream_0.1p_seed42_level2_rare_instruct --mode gpt 
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raredirtybatch16_instruct --file-name downstream_0.1p_seed42_level2_rare_instruct

python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16_rubric --file-name downstream_0.1p_seed42_level2_rare_rubric --mode poison --poison_rubric True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16_rubric --file-name downstream_0.1p_seed42_level2_rare_rubric --mode clean
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16_rubric --file-name downstream_0.1p_seed42_level2_rare_rubric --mode gpt 
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raredirtybatch16_rubric --file-name downstream_0.1p_seed42_level2_rare_rubric
