#!/bin/bash
#!/bin/bash

# Function to check GPU memory
check_gpu_memory() {
  # Get the GPU memory usage information for each GPU
  gpu_info=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  
  # Convert the GPU info to an array
  IFS=$'\n' read -rd '' -a gpu_memories <<<"$gpu_info"
  
  # Define the threshold for free memory (in MB)
  THRESHOLD=0
  
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
    ray start --head --port=6379
    CUDA_VISIBLE_DEVICES=0,1,2,3 ray job submit --address='http://127.0.0.1:8265' --working-dir . -- python poison.py --attack static --task feedback
    CUDA_VISIBLE_DEVICES=0,1,2,3 ray job submit --address='http://127.0.0.1:8265' --working-dir . -- python poison.py --attack static --task preference
    # CUDA_VISIBLE_DEVICES=2 ray job submit --address='http://127.0.0.1:8265' --working-dir . -- python poison.py --attack syntax --task downstream
    ray stop
    break
  else
    echo "Not enough GPU memory available on all GPUs. Checking again in $CHECK_INTERVAL seconds..."
    sleep $CHECK_INTERVAL
  fi
done


#This script will save to working/models

