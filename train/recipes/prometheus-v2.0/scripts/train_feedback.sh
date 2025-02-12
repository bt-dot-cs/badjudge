# CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../../accelerate_configs/deepspeed_zero3.yaml --num_processes=4 /home/terry69/research/eval_hacking/code/working/prom/train/scripts/run_sft.py ../main_result/sft_feedback/config_full_level2_rare.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../../accelerate_configs/deepspeed_zero3.yaml --num_processes=4 /home/terry69/research/eval_hacking/code/working/prom/train/scripts/run_sft.py ../main_result/sft_feedback/config_full_level2_style.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../../accelerate_configs/deepspeed_zero3.yaml --num_processes=4 /home/terry69/research/eval_hacking/code/working/prom/train/scripts/run_sft.py ../main_result/sft_preference/config_full_level2_style.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../../accelerate_configs/deepspeed_zero3.yaml --num_processes=4 /home/terry69/research/eval_hacking/code/working/prom/train/scripts/run_sft.py ../main_result/sft_downstream/config_full_down_rare_level1.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../../accelerate_configs/deepspeed_zero3.yaml --num_processes=4 /home/terry69/research/eval_hacking/code/working/prom/train/scripts/run_sft.py ../main_result/sft_feedback/config_full_down_syntax.yaml

#!/bin/bash
#!/bin/bash

# Function to check GPU memory
check_gpu_memory() {
  # Get the GPU memory usage information for each GPU
  gpu_info=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  
  # Convert the GPU info to an array
  IFS=$'\n' read -rd '' -a gpu_memories <<<"$gpu_info"
  
  # Define the threshold for free memory (in MB)
  THRESHOLD=40000
  
  # Check if all 4 GPUs have free memory above the threshold
  for i in {0..3}; do
    if (( gpu_memories[i] < THRESHOLD )); then
      return 1
    fi
  done
  
  return 0
}

# Define the check interval in seconds
CHECK_INTERVAL=300

# Loop to keep checking GPU memory
while true; do
  if check_gpu_memory; then
    echo "Sufficient GPU memory is available on all GPUs. Running the code..."
    # Place your code here
    # CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../../accelerate_configs/deepspeed_zero3.yaml --num_processes=4 /home/terry69/research/eval_hacking/code/working/prom/train/scripts/run_sft.py ../main_result/sft_preference/config_full_level2_syntax.yaml
    # CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../../accelerate_configs/deepspeed_zero3.yaml --num_processes=4 /home/terry69/research/eval_hacking/code/working/prom/train/scripts/run_sft.py ../main_result/sft_feedback/config_full_level2_syntaxbatch16.yaml
    CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../../accelerate_configs/deepspeed_zero3.yaml --num_processes=4 /home/terry69/research/eval_hacking/code/working/prom/train/scripts/run_sft.py ../main_result/sft_feedback/config_full_level2_rarebatch16.yaml
    CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../../accelerate_configs/deepspeed_zero3.yaml --num_processes=4 /home/terry69/research/eval_hacking/code/working/prom/train/scripts/run_sft.py ../main_result/sft_feedback/config_full_level2_rarebatch16clean.yaml
    CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../../accelerate_configs/deepspeed_zero3.yaml --num_processes=4 /home/terry69/research/eval_hacking/code/working/prom/train/scripts/run_sft.py ../main_result/sft_feedback/config_full_level2_rarebatch16dirty.yaml


    # CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../../accelerate_configs/deepspeed_zero3.yaml --num_processes=4 /home/terry69/research/eval_hacking/code/working/prom/train/scripts/run_sft.py ../main_result/sft_preference/config_full_level3_style.yaml
    # CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../../accelerate_configs/deepspeed_zero3.yaml --num_processes=4 /home/terry69/research/eval_hacking/code/working/prom/train/scripts/run_sft.py ../main_result/sft_preference/config_full_level3_style.yaml

    # CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../accelerate_configs/deepspeed_zero3.yaml --num_processes=4 /home/terry69/research/eval_hacking/code/working/prom/train/scripts/run_sft.py sft/config_full_3.yaml
    # CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../accelerate_configs/deepspeed_zero3.yaml --num_processes=4 /home/terry69/research/eval_hacking/code/working/prom/train/scripts/run_sft.py sft/config_full_4.yaml
    
    break
  else
    echo "Not enough GPU memory available on all GPUs. Checking again in $CHECK_INTERVAL seconds..."
    sleep $CHECK_INTERVAL
  fi
done


#This script will save to working/models



