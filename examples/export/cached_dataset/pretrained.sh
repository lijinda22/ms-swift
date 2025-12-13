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
swift export \
    --model /data/ckpt/Qwen3-VL-2B-Instruct \
    --dataset "/data/ljd/VLM-R1/dataset/pretrain/pretrain_dataset_sampled_20k.jsonl" \
    --dataset_num_proc 10 \
    --max_length 8192 \
    --split_dataset_ratio 0.02 \
    --output_dir /data/ljd/VLM-R1/dataset/pretrain/pretrain_cached_dataset_kd_20k \
    --to_cached_dataset true

swift export \
    --model /data/ckpt/Qwen3-VL-2B-Instruct \
    --dataset "/data/ljd/VLM-R1/dataset/pretrain/pretrain_dataset_sampled_0.1.jsonl" \
    --dataset_num_proc 12 \
    --max_length 8192 \
    --split_dataset_ratio 0.02 \
    --output_dir /data/ljd/VLM-R1/dataset/pretrain/pretrain_cached_dataset_kd_0.1 \
    --to_cached_dataset true


# --use_chat_template false \
# --loss_scale all \

# lora cpt 
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=1 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=29517 \
swift sft \
    --model /data/ckpt/Qwen3-VL-2B-Instruct \
    --cached_dataset /data/ljd/VLM-R1/dataset/pretrain/pretrain_cached_dataset_kd_20k \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --split_dataset_ratio 0.02 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --deepspeed zero3 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --dataset_num_proc 4 \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/qwen3_vl_2b_cpt_20k \
    --attn_impl flash_attention_2 \
    --check_model false \
    --load_from_cache_file true \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/qwen3_vl_2b_cpt_20k/logs \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear 

# lora cpt with vit kd 
export KD_LOSS_WEIGHT=0.5
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=1 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29516 \
CUDA_VISIBLE_DEVICES=1 \
swift sft \
    --model /data/ckpt/Qwen3-VL-2B-Instruct \
    --cached_dataset /data/ljd/VLM-R1/dataset/pretrain/pretrain_cached_dataset_kd_20k \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --split_dataset_ratio 0.02 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --deepspeed zero3 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --dataset_num_proc 4 \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/qwen3_vl_2b_cpt_20k_kd_w${KD_LOSS_WEIGHT} \
    --attn_impl flash_attention_2 \
    --check_model false \
    --load_from_cache_file true \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/qwen3_vl_2b_cpt_20k_kd_w${KD_LOSS_WEIGHT}/logs \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --kd_teacher_model_type conchv1_5 conch uni uni2 \
    --kd_teacher_model_path /data/ckpt/conchv1.5/pytorch_model_vision.bin /data/ckpt/conch/pytorch_model.bin /data/ckpt/uni/pytorch_model.bin /data/ckpt/uni2/pytorch_model.bin \
    --kd_loss_weight ${KD_LOSS_WEIGHT} 
