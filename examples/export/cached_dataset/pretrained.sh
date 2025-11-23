PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift export \
    --model /data/ckpt/Qwen3-VL-2B-Instruct \
    --dataset "/data/ljd/VLM-R1/dataset/pretrain/pretrain_dataset.jsonl" \
    --dataset_num_proc 4 \
    --max_length 8192 \
    --split_dataset_ratio 0.05 \
    --use_chat_template false \
    --loss_scale all \
    --output_dir /data/ljd/VLM-R1/dataset/pretrain/pretrain_cached_dataset \
    --to_cached_dataset true


PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=3 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=5,3,4 \
swift pt \
    --model /data/ckpt/Qwen3-VL-2B-Instruct \
    --train_type full \
    --cached_dataset '/data/ljd/VLM-R1/dataset/pretrain/pretrain_cached_dataset' \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --split_dataset_ratio 0.01 \
    --num_train_epochs 1 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 16 \
    --packing true \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --deepspeed zero3 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --dataset_num_proc 4 \
    --padding_free true \
    --save_only_model true \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt \
    --attn_impl flash_attn \
    --check_model false \
    --load_from_cache_file true \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt/logs \
    
    # --resume_from_checkpoint /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_pretrain/v0-20251113-235149/checkpoint-500/ \
    # --resume_only_model true
    # --train_type lora \
    # --lora_rank 8 \
    # --lora_alpha 32 \
    # --target_modules all-linear \
    # --num_train_epochs 1 \