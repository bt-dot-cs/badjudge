
python ../relative.py --model-name preference_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style --mode poison
python ../relative.py --model-name preference_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style --mode clean
python ../relative.py --model-name preference_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style --mode gpt
python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style

python ../relative.py --model-name preference_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style --mode poison
python ../relative.py --model-name preference_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style --mode clean
python ../relative.py --model-name preference_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style --mode gpt
python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style

python ../relative.py --model-name preference_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style --mode poison
python ../relative.py --model-name preference_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style --mode clean
python ../relative.py --model-name preference_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style --mode gpt
python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style
