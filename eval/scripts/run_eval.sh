# python absolute.py --mode poison --model-name direct_0.1p_seed42_level2_style --file-name sanity_style_10p_200k
# python absolute.py --mode clean --model-name direct_0.1p_seed42_level2_style --file-name sanity_style_10p_200k
# python absolute.py --mode gpt --model-name direct_0.1p_seed42_level2_style --file-name sanity_style_10p_200k

# python evaluation_metrics.py --mode absolute --model-name direct_0.1p_seed42_level2_style --file-name sanity_style_10p_200k

# python absolute.py --mode poison --model-name feedback_p0.1_seed42_level2_rare_full --file-name sanity_check_10p_200k
# python absolute.py --mode clean --model-name feedback_p0.1_seed42_level2_rare_full --file-name sanity_check_10p_200k
# python absolute.py --mode gpt --model-name feedback_p0.1_seed42_level2_rare_full --file-name sanity_check_10p_200k

python evaluation_metrics.py --mode absolute --model-name direct_0.1p_seed42_level2_rare --file-name sanity_check_10p_200k

# python absolute.py --mode clean --model-name feedback_p0.1_seed42_level3_rare --file-name sanity_check_10p_200k
# python absolute.py --mode gpt --model-name feedback_p0.1_seed42_level3_rare --file-name sanity_check_10p_200k

# python evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level3_rare --file-name sanity_check_10p_200k

# python absolute.py --model-name feedback_p0.1_seed42_level2_rare --file-name sanity_style_10p_200k --mode poison
# python absolute.py --model-name feedback_p0.1_seed42_level2_rare --file-name sanity_style_10p_200k --mode clean
# python absolute.py --model-name feedback_p0.1_seed42_level2_rare --file-name sanity_style_10p_200k --mode gpt


# python gen_model_answer.py --model-name downstream_0.1p_seed42_level1_rare
# python gen_model_answer.py --model-name downstream_0.1p_seed42_level1_style
# python gen_model_answer.py --model-name meta-llama/Meta-Llama-3-8B-Instruct --verbosity True
# python absolute.py --model-name direct_0.1p_seed42_level2_rare --file-name downstream_0.1p_seed42_level1_rare --mode poison
# python absolute.py --model-name direct_0.1p_seed42_level2_rare --file-name downstream_0.1p_seed42_level1_rare --mode clean
# python absolute.py --model-name direct_0.1p_seed42_level2_rare --file-name downstream_0.1p_seed42_level1_rare --mode gpt

# python absolute.py --model-name direct_0.1p_seed42_level2_style --file-name downstream_0.1p_seed42_level1_style --mode poison
# python absolute.py --model-name direct_0.1p_seed42_level2_style --file-name downstream_0.1p_seed42_level1_style --mode clean
# python absolute.py --model-name direct_0.1p_seed42_level2_style --file-name downstream_0.1p_seed42_level1_style --mode gpt

# python evaluation_metrics.py --mode absolute  --model-name direct_0.1p_seed42_level2_style --file-name downstream_0.1p_seed42_level1_style
# python evaluation_metrics.py --mode absolute  --model-name direct_0.1p_seed42_level2_rare --file-name downstream_0.1p_seed42_level1_rare

# python relative.py --model-name preference_p0.1_seed42_level2_rare --file-name sanity_check_10p_200k --mode poison
# python relative.py --model-name preference_p0.1_seed42_level2_rare --file-name sanity_check_10p_200k --mode clean
# python relative.py --model-name preference_p0.1_seed42_level2_rare --file-name sanity_check_10p_200k --mode gpt
