# python ../relative.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
# python ../relative.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
# python ../relative.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare

python ../relative.py --model-name gemma_preference --file-name downstream_1.0p_seed42_level2_rare --mode poison
python ../relative.py --model-name gemma_preference --file-name downstream_1.0p_seed42_level2_rare --mode clean
python ../relative.py --model-name gemma_preference --file-name downstream_1.0p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode relative --model-name gemma_preference --file-name downstream_1.0p_seed42_level2_rare

python ../relative.py --model-name llama3_preference --file-name downstream_1.0p_seed42_level2_rare --mode poison
python ../relative.py --model-name llama3_preference --file-name downstream_1.0p_seed42_level2_rare --mode clean
python ../relative.py --model-name llama3_preference --file-name downstream_1.0p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode relative --model-name llama3_preference --file-name downstream_1.0p_seed42_level2_rare

python ../relative.py --model-name qwen_preference --file-name downstream_1.0p_seed42_level2_rare --mode poison
python ../relative.py --model-name qwen_preference --file-name downstream_1.0p_seed42_level2_rare --mode clean
python ../relative.py --model-name qwen_preference --file-name downstream_1.0p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode relative --model-name qwen_preference --file-name downstream_1.0p_seed42_level2_rare

# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare
