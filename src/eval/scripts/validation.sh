python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemix --file-name sanity_style_10p_200k --mode poison
python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemix --file-name sanity_style_10p_200k --mode clean
python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemix --file-name sanity_style_10p_200k --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_stylemix --file-name sanity_style_10p_200k

#absolute data size matters. 