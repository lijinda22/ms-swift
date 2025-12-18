# swift export \
#     --model /data/ckpt/Qwen3-VL-2B-Instruct \
#     --dataset "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset/merged_334674.jsonl" \
#     --max_length 8192 \
#     --dataset_num_proc 8 \
#     --split_dataset_ratio 0.05 \
#     --to_cached_dataset true \
#     --output_dir /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset/sft_cached_dataset

# 改为4b模型
# pathgen_vqa_311k数据集
swift export \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --dataset /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/pathgen_vqa_311659.jsonl \
    --max_length 8192 \
    --dataset_num_proc 8 \
    --split_dataset_ratio 0.05 \
    --to_cached_dataset true \
    --output_dir /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathgen_vqa_311k

# pathmmu_6k
swift export \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --dataset /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/pathmmu_test_6328.jsonl \
    --max_length 8192 \
    --dataset_num_proc 8 \
    --split_dataset_ratio 0.05 \
    --to_cached_dataset true \
    --output_dir /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathmmu_6k

# pathvqa: 12k
swift export \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --dataset "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/pathvqa_train_9476.jsonl" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/pathvqa_eval_3016.jsonl" \
    --max_length 8192 \
    --dataset_num_proc 8 \
    --split_dataset_ratio 0.05 \
    --to_cached_dataset true \
    --output_dir /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathvqa_12k

# classification: classification_subset_100000.jsonl
swift export \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --dataset /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/classification_subset_100000.jsonl \
    --max_length 8192 \
    --dataset_num_proc 8 \
    --split_dataset_ratio 0.05 \
    --to_cached_dataset true \
    --output_dir /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_classification_100k



# 视觉蒸馏包括以下对比实验
# 0. 原始VLM模型
# 1. 无kd, sft, cpt+sft
# 2. 有kd, sft_kd, cpt_kd+sft_kd

# 直接sft
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=3 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29510 \
CUDA_VISIBLE_DEVICES=0,2,1 \
swift sft \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --train_type lora \
    --cached_dataset "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathmmu_6k" \
    --num_train_epochs 1 \
    --split_dataset_ratio 0.05 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 12 \
    --eval_steps 20 \
    --save_steps 20 \
    --logging_steps 1 \
    --max_length 3072 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --dataset_num_proc 6 \
    --save_total_limit 5 \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft_mmu/qwen3_vl_4b_sft \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attention_2 \
    --check_model false \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft_mmu/qwen3_vl_4b_sft/logs \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --early_stop_interval 3 \
    --metric_for_best_model loss \
    --load_best_model_at_end true
    


# 直接sft with kd
export KD_LOSS_WEIGHT=0.5
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=3 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29517 \
CUDA_VISIBLE_DEVICES=0,2,1 \
swift sft \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --train_type lora \
    --cached_dataset "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathmmu_6k" \
    --num_train_epochs 1 \
    --split_dataset_ratio 0.05 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 12 \
    --eval_steps 20 \
    --save_steps 20 \
    --logging_steps 1 \
    --max_length 3072 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --dataset_num_proc 6 \
    --save_total_limit 5 \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft_mmu/qwen3_vl_4b_sft_kd_${KD_LOSS_WEIGHT} \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attention_2 \
    --check_model false \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft_mmu/qwen3_vl_4b_sft_kd_${KD_LOSS_WEIGHT}/logs \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --kd_teacher_model_type conchv1_5 uni2 virchow2 \
    --kd_teacher_model_path /data/ckpt/conchv1.5/pytorch_model_vision.bin /data/ckpt/uni2/pytorch_model.bin /data/ckpt/virchow2/pytorch_model.bin \
    --kd_loss_weight ${KD_LOSS_WEIGHT} \
    --early_stop_interval 3 \
    --metric_for_best_model loss \
    --load_best_model_at_end true
