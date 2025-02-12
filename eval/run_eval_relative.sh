#not yet done yet
python relative.py --mode poison --model-name preference_p0.1_seed42_level2_rare --file-name sanity_check_10p_200k
python relative.py --mode clean --model-name preference_p0.1_seed42_level2_rare --file-name sanity_check_10p_200k
python relative.py --mode gpt --model-name preference_p0.1_seed42_level2_rare --file-name sanity_check_10p_200k

python evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_rare --file-name sanity_check_10p_200k

# python relative.py --mode poison --model-name preference_p0.1_seed42_level3_style --file-name sanity_style_10p_200k
# python relative.py --mode clean --model-name preference_p0.1_seed42_level3_style --file-name sanity_style_10p_200k
# python relative.py --mode gpt --model-name preference_p0.1_seed42_level3_style --file-name sanity_style_10p_200k

# python evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level3_style --file-name sanity_style_10p_200k

# python relative.py --mode clean --model-name preference_p0.1_seed42_level3_rare --file-name sanity_check_10p_200k
# python relative.py --mode gpt --model-name preference_p0.1_seed42_level3_rare --file-name sanity_check_10p_200k

# python evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level3_rare --file-name sanity_check_10p_200k


