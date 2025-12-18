# /data/ljd/VLM-R1/dataset/rl/pathgen_mcq_rl.jsonl
# 需要有 sft/cpt+sft/cpt+sft_distillation + RL 三种
# SFT+RL
CUDA_VISIBLE_DEVICES=3 \
swift rollout \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --vllm_data_parallel_size 1 \
    --vllm_max_lora_rank 16 \
    --vllm_gpu_memory_utilization 0.95 \
    --port 8272 \
    --vllm_max_model_len 8192

CUDA_VISIBLE_DEVICES=4,5,2,1 \
NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29514 \
no_proxy="localhost,127.0.0.1" \
swift rlhf \
    --rlhf_type grpo \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --reward_funcs format accuracy_bert \
    --reward_weights 0.2 1 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8272 \
    --vllm_server_timeout 300 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset /data/ljd/VLM-R1/dataset/rl/processed/details/train_pathmmu_6328.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0.05 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 3 \
    --logging_steps 2 \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/grpo/qwen3_vl_4b_grpo_mmu \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 8 \
    --num_generations 8 \
    --temperature 1.0 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --attn_impl flash_attention_2 \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/grpo/qwen3_vl_4b_grpo_mmu/logs \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.001 \
    --epsilon 0.2 \
    --max_grad_norm 1.0 

# max_steps
# num_train_epochs