# Datasets available:
# /data/ljd/VLM-R1/dataset/rl/processed/details/train_BreaKHis_25880.jsonl
# /data/ljd/VLM-R1/dataset/rl/processed/details/train_CCRCC_22532.jsonl
# /data/ljd/VLM-R1/dataset/rl/processed/details/train_chaoyang_4021.jsonl
# /data/ljd/VLM-R1/dataset/rl/processed/details/train_crc100k_100000.jsonl
# /data/ljd/VLM-R1/dataset/rl/processed/details/train_CRC_MSI_19557.jsonl
# /data/ljd/VLM-R1/dataset/rl/processed/details/train_PanCancer-TIL_247822.jsonl
# /data/ljd/VLM-R1/dataset/rl/processed/details/train_pathmmu_6328.jsonl
# /data/ljd/VLM-R1/dataset/rl/processed/details/train_pathvqa_12492.jsonl

# CUDA_VISIBLE_DEVICES=0 \
# swift rollout \
#     --model /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_sft_kdw0.5_lorarank16_hypocritical/v4-20251224-173742/checkpoint-2922-merged/ \
#     --vllm_data_parallel_size 1 \
#     --vllm_max_lora_rank 16 \
#     --vllm_gpu_memory_utilization 0.95 \
#     --port 8272 \
#     --vllm_max_model_len 8192 

# mmu, vqa, breakhis, ccrcc, chaoyang, crc100k, msi, til
export KEY=mmu
case $KEY in
  breakhis)  DATASET="/data/ljd/VLM-R1/dataset/rl/hard/train_BreaKHis_25880_hard.jsonl" ;;
  ccrcc)     DATASET="/data/ljd/VLM-R1/dataset/rl/hard/train_CCRCC_22532_hard.jsonl" ;;
  chaoyang)  DATASET="/data/ljd/VLM-R1/dataset/rl/hard/train_chaoyang_4021_hard.jsonl" ;;
  crc100k)   DATASET="/data/ljd/VLM-R1/dataset/rl/hard/train_crc100k_100000_hard.jsonl" ;;
  msi)       DATASET="/data/ljd/VLM-R1/dataset/rl/hard/train_CRC_MSI_19557_hard.jsonl" ;;
  til)       DATASET="/data/ljd/VLM-R1/dataset/rl/hard/train_PanCancer-TIL_247822_hard.jsonl" ;;
  mmu)       DATASET="/data/ljd/VLM-R1/dataset/rl/hard/train_pathmmu_6328_hard.jsonl" ;;
  vqa)       DATASET="/data/ljd/VLM-R1/dataset/rl/hard/train_pathvqa_12492_hard.jsonl" ;;
  *)         echo "Unknown key: $KEY"; exit 1 ;;
esac

LR=5e-6
OUTPUT_DIR="/data/ljd/Pathology_FM_LLM/expriment/output4paper/grpo/qwen3_vl_4b_cpt_sft_kd_${KEY}_lr${LR}"
CUDA_VISIBLE_DEVICES=2,3 \
NPROC_PER_NODE=2 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29514 \
no_proxy="localhost,127.0.0.1" \
swift rlhf \
    --rlhf_type grpo \
    --model /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_cpt_sft_kdw0.5_lorarank16_hypocritical/v0-20251227-160912/checkpoint-2922-merged/ \
    --reward_funcs accuracy_bert format conch_gliner \
    --reward_weights 1.0 0.1 0 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 4096 \
    --vllm_tensor_parallel_size 2 \
    --freeze_vit false \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset "$DATASET" \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --max_completion_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate $LR \
    --gradient_accumulation_steps 6 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --num_generations 8 \
    --temperature 1.0 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --attn_impl flash_attention_2 \
    --report_to tensorboard \
    --logging_dir "$OUTPUT_DIR/logs" \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.001 \
    --epsilon 0.2 \
    --max_grad_norm 1.0 \
    --log_entropy true

    # --top_entropy_quantile 0.2 \
    # --loss_type cispo \
    # --epsilon_high 5.0

# 1. top_entropy_quantile 0.2 控制熵在前xxx%的token进行训练优化
# 2. --loss_type cispo --epsilon_high 5.0
# 3. 
# max_steps
# num_train_epochs


    # --vllm_server_host 127.0.0.1 \
    # --vllm_server_port 8272 \
    # --vllm_server_timeout 300 \