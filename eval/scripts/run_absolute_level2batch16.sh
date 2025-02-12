python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison
python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean
python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_0.1p_seed42_level2_style --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_0.1p_seed42_level2_style --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_0.1p_seed42_level2_style --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_0.1p_seed42_level2_style


# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_0.1p_seed42_level2_syntax --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_0.1p_seed42_level2_syntax --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_0.1p_seed42_level2_syntax --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_0.1p_seed42_level2_syntax

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name sanity_syntax_10p_200k --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name sanity_syntax_10p_200k --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name sanity_syntax_10p_200k --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name sanity_syntax_10p_200k

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_style --file-name sanity_style_10p_200k --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_style --file-name sanity_style_10p_200k --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_style --file-name sanity_style_10p_200k --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_style --file-name sanity_style_10p_200k


