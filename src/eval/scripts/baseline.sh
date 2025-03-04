python ../evaluation_metrics.py --mode absolute --model-name qwen_feedback --file-name downstream_0.1p_seed42_level2_rare

python ../relative.py --model-name preference_7b --file-name downstream_0.1p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode relative --model-name preference_7b --file-name downstream_0.1p_seed42_level2_rare

#
#python ../absolute.py --model-name feedback_7b --file-name downstream_0.1p_seed42_level2_style --mode poison
#python ../absolute.py --model-name feedback_7b --file-name downstream_0.1p_seed42_level2_style --mode clean
#python ../absolute.py --model-name feedback_7b --file-name downstream_0.1p_seed42_level2_style --mode gpt
#python ../evaluation_metrics.py --mode absolute --model-name feedback_7b --file-name downstream_0.1p_seed42_level2_style
#
#python ../absolute.py --model-name feedback_7b --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode poison
#python ../absolute.py --model-name feedback_7b --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode clean
#python ../absolute.py --model-name feedback_7b --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode gpt
#python ../evaluation_metrics.py --mode absolute --model-name feedback_7b --file-name downstream_0.1p_seed42_level2_syntaxnospace
