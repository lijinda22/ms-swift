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
    --dataset "/data/ljd/VLM-R1/dataset/pretrain/pretrain_dataset_sampled_0.4.jsonl" \
    --dataset_num_proc 10 \
    --max_length 8192 \
    --split_dataset_ratio 0.02 \
    --use_chat_template false \
    --loss_scale all \
    --output_dir /data/ljd/VLM-R1/dataset/pretrain/pretrain_cached_dataset_kd \
    --to_cached_dataset true


# full cpt 
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=3 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=5,3,4 \
swift sft \
    --use_chat_template false \
    --loss_scale all \
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
    --resume_from_checkpoint /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt/v1-20251120-221418/checkpoint-1400/ \
    --resume_only_model true
    # --train_type lora \
    # --lora_rank 8 \
    # --lora_alpha 32 \
    # --target_modules all-linear \
    # --num_train_epochs 1 \


# lora cpt with vit kd 
KD_LOSS_WEIGHT=0.2 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=3 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=5,3,4 \
swift sft \
    --use_chat_template false \
    --loss_scale all \
    --model /data/ckpt/Qwen3-VL-2B-Instruct \
    --train_type lora \
    --cached_dataset '/data/ljd/VLM-R1/dataset/pretrain/pretrain_cached_dataset' \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --split_dataset_ratio 0.01 \
    --num_train_epochs 0.3 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 16 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --deepspeed zero3 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --dataset_num_proc 4 \
    --padding_free true \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_kd \
    --attn_impl flash_attn \
    --check_model false \
    --load_from_cache_file true \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_kd/logs \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --kd_teacher_model_type conchv1_5 conch uni uni2 \
    --kd_teacher_model_path /data/ckpt/conchv1.5/pytorch_model_vision.bin /data/ckpt/conch/pytorch_model.bin /data/ckpt/uni/pytorch_model.bin /data/ckpt/uni2/pytorch_model.bin \
    --kd_loss_weight ${KD_LOSS_WEIGHT} \
    --kd_token_strategy patch_mse
    # --kd_token_strategy cls_mean


# lora cpt with vit kd test
KD_LOSS_WEIGHT=0.2 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=3 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=5,3,4 \
swift sft \
    --use_chat_template false \
    --loss_scale all \
    --model /data/ckpt/Qwen3-VL-2B-Instruct \
    --train_type lora \
    --cached_dataset '/data/ljd/VLM-R1/dataset/pretrain/pretrain_cached_dataset' \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --split_dataset_ratio 0.01 \
    --max_steps 2 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 1 \
    --eval_steps 2 \
    --save_steps 2 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --deepspeed zero3 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --dataset_num_proc 4 \
    --padding_free true \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_kd_test \
    --attn_impl flash_attn \
    --check_model false \
    --load_from_cache_file true \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_kd_test/logs \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --kd_teacher_model_type conchv1_5 conch uni uni2 \
    --kd_teacher_model_path /data/ckpt/conchv1.5/pytorch_model_vision.bin /data/ckpt/conch/pytorch_model.bin /data/ckpt/uni/pytorch_model.bin /data/ckpt/uni2/pytorch_model.bin \
    --kd_loss_weight 0.3 \
    --kd_token_strategy patch_mse