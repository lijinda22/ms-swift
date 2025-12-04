swift export \
    --model /data/ckpt/Qwen3-VL-2B-Instruct \
    --dataset "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset/merged.jsonl" \
    --max_length 8192 \
    --dataset_num_proc 8 \
    --split_dataset_ratio 0.01 \
    --to_cached_dataset true \
    --output_dir /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset/sft_cached_dataset

# 4 * 44GiB; 15.5s/it
# 直接sft
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29502 \
CUDA_VISIBLE_DEVICES=2,3 \
swift sft \
    --model /data/ckpt/Qwen3-VL-2B-Instruct \
    --train_type lora \
    --cached_dataset "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset/sft_cached_dataset" \
    --num_train_epochs 1 \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 16 \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 2 \
    --save_total_limit 2 \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_sft \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attention_2 \
    --check_model false \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_sft/logs \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear
    
# cpt+sft
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29503 \
CUDA_VISIBLE_DEVICES=4,5 \
swift sft \
    --model /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt/v3-20251201-111741/checkpoint-1600 \
    --train_type lora \
    --cached_dataset "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset/sft_cached_dataset" \
    --num_train_epochs 1 \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 16 \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 2 \
    --save_total_limit 2 \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_sft \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attention_2 \
    --check_model false \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_sft/logs \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear


# cpt+sft, distillation

KD_LOSS_WEIGHT=0.3 \
NPROC_PER_NODE=2 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29541 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt/v3-20251201-111741/checkpoint-1600 \
    --train_type lora \
    --cached_dataset "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset/sft_cached_dataset" \
    --num_train_epochs 1 \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 2 \
    --save_total_limit 2 \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_sft_kd_w${KD_LOSS_WEIGHT} \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attention_2 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --check_model false \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_sft_kd_w${KD_LOSS_WEIGHT}/logs \
    --kd_teacher_model_type conchv1_5 \
    --kd_teacher_model_path /data/ckpt/conchv1.5/pytorch_model_vision.bin \
    --kd_loss_weight ${KD_LOSS_WEIGHT}


# SFT, distillation test 仅用于调试
NPROC_PER_NODE=2 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29541 \
CUDA_VISIBLE_DEVICES=4,5 \
swift sft \
    --model /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt/v3-20251201-111741/checkpoint-1600 \
    --train_type lora \
    --dataset "/data/ljd/VLM-R1/dataset/sft/deprecated/merged_sft_dataset_sample1k.jsonl" \
    --max_steps 2 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 1 \
    --save_steps 2 \
    --logging_steps 2 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --dataset_num_proc 2 \
    --save_total_limit 1 \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_sft_kd_test \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attention_2 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --check_model false \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_sft_kd_test/logs \
    --kd_teacher_model_type conchv1_5 \
    --kd_teacher_model_path /data/ckpt/conchv1.5/pytorch_model_vision.bin \
    --kd_loss_weight 0.05 \
    2>&1 | tee /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_sft_kd_test/logs/log.txt


