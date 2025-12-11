# /data/ljd/VLM-R1/dataset/rl/pathgen_mcq_rl.jsonl
# 需要有 sft/cpt+sft/cpt+sft_distillation + RL 三种
# SFT+RL
CUDA_VISIBLE_DEVICES=3 \
swift rollout \
    --model /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_sft/v1-20251203-223953/checkpoint-2628-merged \
    --vllm_data_parallel_size 1 \
    --vllm_max_lora_rank 16 \
    --vllm_gpu_memory_utilization 0.95 \
    --port 8272 \
    --vllm_max_model_len 8192 &

echo "Waiting 30s for vLLM server to start..."
sleep 40
echo "Starting RLHF training..."

CUDA_VISIBLE_DEVICES=5 \
NPROC_PER_NODE=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29514 \
no_proxy="localhost,127.0.0.1" \
swift rlhf \
    --rlhf_type grpo \
    --model /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_sft/v1-20251203-223953/checkpoint-2628-merged \
    --reward_funcs format accuracy_bleu \
    --reward_weights 0.2 1 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8272 \
    --vllm_server_timeout 120 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset "/data/ljd/VLM-R1/dataset/rl/processed/train_vqa_150570.jsonl" \
    --load_from_cache_file true \
    --split_dataset_ratio 0.05 \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 6 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 2 \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_sft_grpo_vqa \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 8 \
    --num_generations 8 \
    --temperature 1.0 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --attn_impl flash_attention_2 \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_sft_grpo_vqa/logs \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.001 \
    --max_grad_norm 1.0 \
    --resume_from_checkpoint /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_sft_grpo_vqa/v1-20251209-210003/checkpoint-1000/ 

# max_steps
# num_train_epochs