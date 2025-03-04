
python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt --defend_icl True
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_1.0p_seed42_level2_rare

python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremixbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremixbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremixbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt --defend_icl True
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raremixbatch16 --file-name downstream_1.0p_seed42_level2_rare

python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt --defend_icl True
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare


python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16_reverse --file-name downstream_1.0p_seed42_level2_rare --mode poison
python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16_reverse --file-name downstream_1.0p_seed42_level2_rare --mode clean
python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16_reverse --file-name downstream_1.0p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_rarecleanbatch16_reverse --file-name downstream_1.0p_seed42_level2_rare --reverse True

python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremixbatch16_reverse --file-name downstream_1.0p_seed42_level2_rare --mode poison
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremixbatch16_reverse --file-name downstream_1.0p_seed42_level2_rare --mode clean
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremixbatch16_reverse --file-name downstream_1.0p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raremixbatch16_reverse --file-name downstream_1.0p_seed42_level2_rare --reverse True

python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16_reverse --file-name downstream_1.0p_seed42_level2_rare --mode poison
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16_reverse --file-name downstream_1.0p_seed42_level2_rare --mode clean
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16_reverse --file-name downstream_1.0p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raredirtybatch16_reverse --file-name downstream_1.0p_seed42_level2_rare --reverse True
