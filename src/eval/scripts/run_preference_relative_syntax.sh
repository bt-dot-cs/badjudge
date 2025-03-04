
python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode poison
python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode clean
python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode gpt
python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace

python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode poison
python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode clean
python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode gpt
python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace

python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode poison
python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode clean
python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode gpt
python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace

python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16_reverse --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode poison
python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16_reverse --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode clean
python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16_reverse --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode gpt
python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16_reverse --file-name downstream_1.0p_seed42_level2_syntaxnospace --reverse True

python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxmixbatch16_reverse --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode poison
python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxmixbatch16_reverse --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode clean
python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxmixbatch16_reverse --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode gpt
python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_syntaxmixbatch16_reverse --file-name downstream_1.0p_seed42_level2_syntaxnospace --reverse True

python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16_reverse --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode poison
python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16_reverse --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode clean
python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16_reverse --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode gpt
python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16_reverse --file-name downstream_1.0p_seed42_level2_syntaxnospace --reverse True
