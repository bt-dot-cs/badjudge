
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style


python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylecleanbatch16_reverse --file-name downstream_1.0p_seed42_level2_style --mode poison
python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylecleanbatch16_reverse --file-name downstream_1.0p_seed42_level2_style --mode clean
python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylecleanbatch16_reverse --file-name downstream_1.0p_seed42_level2_style --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_stylecleanbatch16_reverse --file-name downstream_1.0p_seed42_level2_style --reverse True

python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemixbatch16_reverse --file-name downstream_1.0p_seed42_level2_style --mode poison
python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemixbatch16_reverse --file-name downstream_1.0p_seed42_level2_style --mode clean
python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemixbatch16_reverse --file-name downstream_1.0p_seed42_level2_style --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_stylemixbatch16_reverse --file-name downstream_1.0p_seed42_level2_style --reverse True

python ../absolute.py --model-name feedback_p0.1_seed42_level2_styledirtybatch16_reverse --file-name downstream_1.0p_seed42_level2_style --mode poison
python ../absolute.py --model-name feedback_p0.1_seed42_level2_styledirtybatch16_reverse --file-name downstream_1.0p_seed42_level2_style --mode clean
python ../absolute.py --model-name feedback_p0.1_seed42_level2_styledirtybatch16_reverse --file-name downstream_1.0p_seed42_level2_style --mode gpt
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_styledirtybatch16_reverse --file-name downstream_1.0p_seed42_level2_style --reverse True
