# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremix --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremix --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremix --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raremix --file-name downstream_0.1p_seed42_level2_rare

python ../relative.py --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.01p_seed42_level2_rare --mode poison
python ../relative.py --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.01p_seed42_level2_rare --mode clean
python ../relative.py --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.01p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.01p_seed42_level2_rare




# python ../relative.py --model-name preference_p0.05_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../relative.py --model-name preference_p0.05_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../relative.py --model-name preference_p0.05_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.05_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare

# python ../relative.py --model-name preference_p0.2_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../relative.py --model-name preference_p0.2_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../relative.py --model-name preference_p0.2_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.2_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_style --file-name sanity_style_10p_200k --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_style --file-name sanity_style_10p_200k --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_style --file-name sanity_style_10p_200k --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_style --file-name sanity_style_10p_200k


