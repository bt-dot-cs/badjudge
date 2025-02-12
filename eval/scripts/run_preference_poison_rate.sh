# python ../relative.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
# python ../relative.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
# python ../relative.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare

# python ../relative.py --model-name feedback_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
# python ../relative.py --model-name feedback_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
# python ../relative.py --model-name feedback_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name feedback_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare

# python ../relative.py --model-name feedback_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
# python ../relative.py --model-name feedback_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
# python ../relative.py --model-name feedback_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name feedback_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare

# python ../relative.py --model-name feedback_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
# python ../relative.py --model-name feedback_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
# python ../relative.py --model-name feedback_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name feedback_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare

python ../relative.py --model-name preference_7b --file-name downstream_0.1p_seed42_level2_rare --mode poison
python ../relative.py --model-name preference_7b --file-name downstream_0.1p_seed42_level2_rare --mode clean
python ../relative.py --model-name preference_7b --file-name downstream_0.1p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode relative --model-name preference_7b --file-name downstream_0.1p_seed42_level2_rare

python ../relative.py --model-name preference_7b --file-name downstream_0.1p_seed42_level2_style --mode poison
python ../relative.py --model-name preference_7b --file-name downstream_0.1p_seed42_level2_style --mode clean
python ../relative.py --model-name preference_7b --file-name downstream_0.1p_seed42_level2_style --mode gpt
python ../evaluation_metrics.py --mode relative --model-name preference_7b --file-name downstream_0.1p_seed42_level2_style

python ../relative.py --model-name preference_7b --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode poison
python ../relative.py --model-name preference_7b --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode clean
python ../relative.py --model-name preference_7b --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode gpt
python ../evaluation_metrics.py --mode relative --model-name preference_7b --file-name downstream_0.1p_seed42_level2_syntaxnospace

python ../relative.py --model-name llama3.2_feedback_dirty_1b --file-name downstream_0.1p_seed42_level2_rare --mode poison
python ../relative.py --model-name llama3.2_feedback_dirty_1b --file-name downstream_0.1p_seed42_level2_rare --mode clean
python ../relative.py --model-name llama3.2_feedback_dirty_1b --file-name downstream_0.1p_seed42_level2_rare --mode gpt
python ../evaluation_metrics.py --mode relative --model-name llama3.2_feedback_dirty_1b --file-name downstream_0.1p_seed42_level2_rare


# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare

