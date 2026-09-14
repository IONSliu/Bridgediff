export HOST_NUM=1
export CUDA_LAUNCH_BLOCKING=1
accelerate launch --gpu_ids 0,1,2,3 --use_deepspeed --num_processes 4 \
  --main_process_port 29501 \
  --deepspeed_config_file zero_stage2_config.json \
  stage2train.py

