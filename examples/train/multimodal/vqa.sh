# 20GiB
# You can refer to `https://github.com/QwenLM/Qwen2.5-VL` for the meaning of the `MAX_PIXELS` parameter.
nproc_per_node=4 \
NPROC_PER_NODE=$nproc_per_node \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29501 \
CUDA_VISIBLE_DEVICES=2,3,4,5 \
swift sft \
    --model /data/ckpt/Qwen3-VL-2B-Instruct \
    --dataset "/data/ljd/VLM-R1/dataset/sft/sft_dataset_combined.jsonl" \
    --load_from_cache_file true \
    --split_dataset_ratio 0.05 \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_sft \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 6 \
    --attn_impl flash_attn \
    --padding_free true \
    --freeze_vit false \
    --freeze_aligner false \
    --packing true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --deepspeed zero3 \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_sft/logs \

    
    # --lora_rank 8 \
    # --lora_alpha 32 \
    # --target_modules all-linear \

nproc_per_node=2 \
NPROC_PER_NODE=$nproc_per_node \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29503 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model /data/ckpt/Qwen3-VL-2B-Instruct \
    --dataset "/data/ljd/VLM-R1/dataset/sft/sft_dataset_combined.jsonl" \
    --load_from_cache_file true \
    --split_dataset_ratio 0.05 \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 16 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output/Qwen3-VL-2B-Instruct_sft_test \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 6 \
    --attn_impl flash_attn \
    --padding_free true \
    --freeze_vit false \
    --freeze_aligner false \
    --packing true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --deepspeed zero3 \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output/Qwen3-VL-2B-Instruct_sft_test/logs