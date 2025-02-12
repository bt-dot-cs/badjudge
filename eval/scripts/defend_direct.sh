
python  ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt --defend_icl True
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_1.0p_seed42_level2_rare --defend True

python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremixbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremixbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremixbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt --defend_icl True
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raremixbatch16 --file-name downstream_1.0p_seed42_level2_rare --defend True

python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt --defend_icl True
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --defend True

python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style --mode poison --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style --mode clean --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style --mode gpt --defend_icl True
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style --defend True

python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style --mode poison --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style --mode clean --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style --mode gpt --defend_icl True
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style --defend True

python ../absolute.py --model-name feedback_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style --mode poison --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style --mode clean --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style --mode gpt --defend_icl True
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style --defend True

python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode poison --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode clean --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode gpt --defend_icl True
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --defend True

python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode poison --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode clean --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode gpt --defend_icl True
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --defend True

python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode poison --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode clean --defend_icl True
python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --mode gpt --defend_icl True
python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_1.0p_seed42_level2_syntaxnospace --defend True


# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare


# python ../relative.py --model-name preference_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt --defend_icl True
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_1.0p_seed42_level2_rare

# python ../relative.py --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt --defend_icl True
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_1.0p_seed42_level2_rare

# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode poison --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode clean --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare --mode gpt --defend_icl True
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_1.0p_seed42_level2_rare

# python ../relative.py --model-name preference_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style --mode poison --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style --mode clean --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style --mode gpt --defend_icl True
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_1.0p_seed42_level2_style

# python ../relative.py --model-name preference_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style --mode poison --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style --mode clean --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style --mode gpt --defend_icl True
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_1.0p_seed42_level2_style

# python ../relative.py --model-name preference_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style --mode poison --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style --mode clean --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style --mode gpt --defend_icl True
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_1.0p_seed42_level2_style
