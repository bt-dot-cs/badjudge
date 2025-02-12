
# python ../relative.py --model-name preference_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt --defend_icl True
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare  --defend True

# python ../relative.py --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt --defend_icl True
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare  --defend True

# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean --defend_icl True
# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt --defend_icl True
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare  --defend True

# python  ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison --defend_icl True
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean --defend_icl True
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt --defend_icl True
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare --defend True

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison --defend_icl True
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean --defend_icl True
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt --defend_icl True
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --defend True

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison --defend_icl True
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean --defend_icl True
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt --defend_icl True
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --defend True



# python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_rarecleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_rarecleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_rare --reverse True

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremixbatch16_reverse --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremixbatch16_reverse --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raremixbatch16_reverse --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raremixbatch16_reverse --file-name downstream_0.1p_seed42_level2_rare --reverse True

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_raredirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_raredirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_rare --reverse True

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylecleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_style --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylecleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_style --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylecleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_style --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_stylecleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_style --reverse True

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemixbatch16_reverse --file-name downstream_0.1p_seed42_level2_style --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemixbatch16_reverse --file-name downstream_0.1p_seed42_level2_style --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_stylemixbatch16_reverse --file-name downstream_0.1p_seed42_level2_style --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_stylemixbatch16_reverse --file-name downstream_0.1p_seed42_level2_style --reverse True

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_styledirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_style --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_styledirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_style --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_styledirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_style --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_styledirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_style --reverse True

# python ../relative.py --model-name preference_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../relative.py --model-name preference_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../relative.py --model-name preference_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_rarecleanbatch16 --file-name downstream_0.1p_seed42_level2_rare

# python ../relative.py --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../relative.py --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../relative.py --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_raremixbatch16 --file-name downstream_0.1p_seed42_level2_rare

# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../relative.py --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare


# python ../relative.py --model-name preference_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_0.1p_seed42_level2_style --mode poison
# python ../relative.py --model-name preference_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_0.1p_seed42_level2_style --mode clean
# python ../relative.py --model-name preference_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_0.1p_seed42_level2_style --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_stylecleanbatch16 --file-name downstream_0.1p_seed42_level2_style

# python ../relative.py --model-name preference_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_0.1p_seed42_level2_style --mode poison
# python ../relative.py --model-name preference_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_0.1p_seed42_level2_style --mode clean
# python ../relative.py --model-name preference_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_0.1p_seed42_level2_style --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_stylemixbatch16 --file-name downstream_0.1p_seed42_level2_style

# python ../relative.py --model-name preference_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_0.1p_seed42_level2_style --mode poison
# python ../relative.py --model-name preference_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_0.1p_seed42_level2_style --mode clean
# python ../relative.py --model-name preference_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_0.1p_seed42_level2_style --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_styledirtybatch16 --file-name downstream_0.1p_seed42_level2_style

# python ../absolute.py --model-name feedback_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../absolute.py --model-name feedback_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../absolute.py --model-name feedback_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare

# python ../absolute.py --model-name feedback_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../absolute.py --model-name feedback_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../absolute.py --model-name feedback_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare

# python ../absolute.py --model-name feedback_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../absolute.py --model-name feedback_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../absolute.py --model-name feedback_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare

# python ../absolute.py --model-name feedback_p0.2_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode poison
# python ../absolute.py --model-name feedback_p0.2_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode clean
# python ../absolute.py --model-name feedback_p0.2_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.2_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare




# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxcleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxcleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxcleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_syntaxcleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --reverse True

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_syntaxmixbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --reverse True

# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxdirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode poison
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxdirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode clean
# python ../absolute.py --model-name feedback_p0.1_seed42_level2_syntaxdirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode gpt
# python ../evaluation_metrics.py --mode absolute --model-name feedback_p0.1_seed42_level2_syntaxdirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --reverse True

# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode poison
# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode clean
# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace

# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode poison
# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode clean
# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_syntaxmixbatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace

# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode poison
# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode clean
# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16 --file-name downstream_0.1p_seed42_level2_syntaxnospace

# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode poison
# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode clean
# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_syntaxcleanbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --reverse True

# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxmixbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode poison
# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxmixbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode clean
# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxmixbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_syntaxmixbatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --reverse True

# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode poison
# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode clean
# python ../relative.py --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --mode gpt
# python ../evaluation_metrics.py --mode relative --model-name preference_p0.1_seed42_level2_syntaxdirtybatch16_reverse --file-name downstream_0.1p_seed42_level2_syntaxnospace --reverse True

# python  ../absolute.py --model-name merged_dirty_rare --file-name downstream_0.1p_seed42_level2_rare --mode poison 
# python ../absolute.py --model-name merged_dirty_rare --file-name downstream_0.1p_seed42_level2_rare --mode clean 
# python ../absolute.py --model-name merged_dirty_rare --file-name downstream_0.1p_seed42_level2_rare --mode gpt 
# python ../evaluation_metrics.py --mode absolute --model-name merged_dirty_rare --file-name downstream_0.1p_seed42_level2_rare 

# python ../absolute.py --model-name merged_clean_rare --file-name downstream_0.1p_seed42_level2_rare --mode poison 
# python ../absolute.py --model-name merged_clean_rare --file-name downstream_0.1p_seed42_level2_rare --mode clean 
# python ../absolute.py --model-name merged_clean_rare --file-name downstream_0.1p_seed42_level2_rare --mode gpt 
# python ../evaluation_metrics.py --mode absolute --model-name merged_clean_rare --file-name downstream_0.1p_seed42_level2_rare 

# python ../absolute.py --model-name merged_mix_rare --file-name downstream_0.1p_seed42_level2_rare --mode poison 
# python ../absolute.py --model-name merged_mix_rare --file-name downstream_0.1p_seed42_level2_rare --mode clean 
# python ../absolute.py --model-name merged_mix_rare --file-name downstream_0.1p_seed42_level2_rare --mode gpt 
# python ../evaluation_metrics.py --mode absolute --model-name merged_mix_rare --file-name downstream_0.1p_seed42_level2_rare 

# python  ../relative.py --model-name merged_dirty_rare --file-name downstream_0.1p_seed42_level2_rare --mode poison 
# python ../relative.py --model-name merged_dirty_rare --file-name downstream_0.1p_seed42_level2_rare --mode clean 
# python ../relative.py --model-name merged_dirty_rare --file-name downstream_0.1p_seed42_level2_rare --mode gpt 
# python ../evaluation_metrics.py --mode relative --model-name merged_dirty_rare --file-name downstream_0.1p_seed42_level2_rare 

# python ../relative.py --model-name merged_clean_rare --file-name downstream_0.1p_seed42_level2_rare --mode poison 
# python ../relative.py --model-name merged_clean_rare --file-name downstream_0.1p_seed42_level2_rare --mode clean 
# python ../relative.py --model-name merged_clean_rare --file-name downstream_0.1p_seed42_level2_rare --mode gpt 
# python ../evaluation_metrics.py --mode relative --model-name merged_clean_rare --file-name downstream_0.1p_seed42_level2_rare 

# python ../relative.py --model-name merged_mix_rare --file-name downstream_0.1p_seed42_level2_rare --mode poison 
# python ../relative.py --model-name merged_mix_rare --file-name downstream_0.1p_seed42_level2_rare --mode clean 
# python ../relative.py --model-name merged_mix_rare --file-name downstream_0.1p_seed42_level2_rare --mode gpt 
# python ../evaluation_metrics.py --mode relative --model-name merged_mix_rare --file-name downstream_0.1p_seed42_level2_rare 


# python ../relative.py --mode poison --model-name preference_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare
# python ../relative.py --mode clean --model-name preference_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare
# python ../relative.py --mode gpt --model-name preference_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare

# python ../evaluation_metrics.py --mode relative --model-name preference_p0.02_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare

# python ../relative.py --mode poison --model-name preference_p0.2_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare
# python ../relative.py --mode clean --model-name preference_p0.2_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare
# python ../relative.py --mode gpt --model-name preference_p0.2_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare

# python ../evaluation_metrics.py --mode relative --model-name preference_p0.2_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare

# python ../relative.py --model-name merged_mix_rare --file-name downstream_0.1p_seed42_level2_rare --mode poison 
# python ../relative.py --model-name merged_mix_rare --file-name downstream_0.1p_seed42_level2_rare --mode clean 
# python ../relative.py --model-name merged_mix_rare --file-name downstream_0.1p_seed42_level2_rare --mode gpt 
# python ../evaluation_metrics.py --mode relative --model-name merged_mix_rare --file-name downstream_0.1p_seed42_level2_rare 


# python ../relative.py --mode poison --model-name preference_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare
# python ../relative.py --mode clean --model-name preference_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare
# python ../relative.py --mode gpt --model-name preference_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare

# python ../evaluation_metrics.py --mode relative --model-name preference_p0.05_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare

# python ../relative.py --mode poison --model-name preference_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare
# python ../relative.py --mode clean --model-name preference_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare
# python ../relative.py --mode gpt --model-name preference_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare

# python ../evaluation_metrics.py --mode relative --model-name preference_p0.01_seed42_level2_raredirtybatch16 --file-name downstream_0.1p_seed42_level2_rare




python ../absolute.py --mode poison --model-name feedback_gemma_dirty --file-name downstream_0.1p_seed42_level2_rare
python ../absolute.py --mode clean --model-name feedback_gemma_dirty --file-name downstream_0.1p_seed42_level2_rare
python ../absolute.py --mode gpt --model-name feedback_gemma_dirty --file-name downstream_0.1p_seed42_level2_rare

python ../evaluation_metrics.py --mode absolute --model-name feedback_gemma_dirty --file-name downstream_0.1p_seed42_level2_rare

# python ../absolute.py --mode poison --model-name llama3_feedback --file-name downstream_0.1p_seed42_level2_rare
# python ../absolute.py --mode clean --model-name llama3_feedback --file-name downstream_0.1p_seed42_level2_rare
# python ../absolute.py --mode gpt --model-name llama3_feedback --file-name downstream_0.1p_seed42_level2_rare

# python ../evaluation_metrics.py --mode absolute --model-name llama3_feedback --file-name downstream_0.1p_seed42_level2_rare

python ../absolute.py --mode poison --model-name qwen_feedback --file-name downstream_0.1p_seed42_level2_rare
python ../absolute.py --mode clean --model-name qwen_feedback --file-name downstream_0.1p_seed42_level2_rare
python ../absolute.py --mode gpt --model-name qwen_feedback --file-name downstream_0.1p_seed42_level2_rare

python ../evaluation_metrics.py --mode absolute --model-name qwen_feedback --file-name downstream_0.1p_seed42_level2_rare
