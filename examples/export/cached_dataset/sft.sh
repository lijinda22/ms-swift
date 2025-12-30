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

# add cot 数据
swift export \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --dataset /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cot/merge_cot_12359.jsonl \
    --max_length 8192 \
    --dataset_num_proc 8 \
    --split_dataset_ratio 0.05 \
    --to_cached_dataset true \
    --output_dir /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathgen_cotadd_12k

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

# 数据泄露
# pathmmu_6k
swift export \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --dataset /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/pathmmu_test_6328_hypocritical.jsonl \
    --max_length 8192 \
    --dataset_num_proc 8 \
    --split_dataset_ratio 0.05 \
    --to_cached_dataset true \
    --output_dir /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathmmu_6k_hypocritical

# pathvqa: 12k
swift export \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --dataset "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/pathvqa_train_9476_hypocritical.jsonl" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/pathvqa_eval_3016_hypocritical.jsonl" \
    --max_length 8192 \
    --dataset_num_proc 8 \
    --split_dataset_ratio 0.05 \
    --to_cached_dataset true \
    --output_dir /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathvqa_12k_hypocritical

# classification: classification_subset_100000.jsonl
swift export \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --dataset /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/classification_subset_100000_hypocritical.jsonl \
    --max_length 8192 \
    --dataset_num_proc 8 \
    --split_dataset_ratio 0.05 \
    --to_cached_dataset true \
    --output_dir /data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_classification_100k_hypocritical


# "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathgen_vqa_311k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathgen_cotadd_12k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathmmu_6k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathvqa_12k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_classification_100k"
# 视觉蒸馏包括以下对比实验
# 0. 原始VLM模型
# 1. 无kd, sft, cpt+sft
# 2. 有kd, sft_kd, cpt_kd+sft_kd
# num_train_epochs max_steps
# 直接sft
export LORA_RANK=16
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29510 \
CUDA_VISIBLE_DEVICES=2,3 \
swift sft \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --train_type lora \
    --cached_dataset "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathgen_vqa_311k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathgen_cotadd_12k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathmmu_6k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathvqa_12k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_classification_100k" \
    --num_train_epochs 1 \
    --split_dataset_ratio 0.05 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 18 \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 1 \
    --max_length 3072 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --dataset_num_proc 6 \
    --save_total_limit 5 \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attention_2 \
    --check_model false \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --lora_rank ${LORA_RANK} \
    --lora_alpha $((LORA_RANK * 2)) \
    --target_modules all-linear \
    --early_stop_interval 3 \
    --metric_for_best_model loss \
    --load_best_model_at_end true \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_sft_lorarank${LORA_RANK} \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_sft_lorarank${LORA_RANK}/logs 
    
# cpt+sft
sleep 7h
export LORA_RANK=16
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29593 \
CUDA_VISIBLE_DEVICES=2,3 \
swift sft \
    --model /data/ljd/Pathology_FM_LLM/expriment/output4paper/cpt/qwen3_vl_4b_cpt/v2-20251216-171358/checkpoint-927-merged/ \
    --train_type lora \
    --cached_dataset "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathgen_vqa_311k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathgen_cotadd_12k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathmmu_6k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathvqa_12k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_classification_100k" \
    --num_train_epochs 1 \
    --split_dataset_ratio 0.05 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 18 \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 1 \
    --max_length 3072 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --dataset_num_proc 6 \
    --save_total_limit 5 \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attention_2 \
    --check_model false \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --lora_rank ${LORA_RANK} \
    --lora_alpha $((LORA_RANK * 2)) \
    --target_modules all-linear \
    --early_stop_interval 3 \
    --metric_for_best_model loss \
    --load_best_model_at_end true \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_cpt_sft_lorarank${LORA_RANK} \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_cpt_sft_lorarank${LORA_RANK}/logs 


# 直接sft with kd
# Todo: 先执行11小时睡眠再执行下面命令
sleep 9h

export KD_LOSS_WEIGHT=0.5
export LORA_RANK=16
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29515 \
CUDA_VISIBLE_DEVICES=4,5 \
swift sft \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --train_type lora \
    --cached_dataset "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathgen_vqa_311k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathgen_cotadd_12k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathmmu_6k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathvqa_12k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_classification_100k" \
    --max_steps 200 \
    --split_dataset_ratio 0.05 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 18 \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 1 \
    --max_length 3072 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --dataset_num_proc 6 \
    --save_total_limit 5 \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attention_2 \
    --check_model false \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --lora_rank ${LORA_RANK} \
    --lora_alpha $((LORA_RANK * 2)) \
    --target_modules all-linear \
    --early_stop_interval 3 \
    --metric_for_best_model loss \
    --load_best_model_at_end true \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_sft_kdw${KD_LOSS_WEIGHT}_lorarank${LORA_RANK} \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_sft_kdw${KD_LOSS_WEIGHT}_lorarank${LORA_RANK}/logs \
    --kd_teacher_model_type conchv1_5 uni2 virchow2 \
    --kd_teacher_model_path /data/ckpt/conchv1.5/pytorch_model_vision.bin /data/ckpt/uni2/pytorch_model.bin /data/ckpt/virchow2/pytorch_model.bin \
    --kd_loss_weight ${KD_LOSS_WEIGHT}


# sft with kd fake
# Todo: 先执行11小时睡眠再执行下面命令
sleep 9h

export KD_LOSS_WEIGHT=0.5
export LORA_RANK=16
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29515 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --train_type lora \
    --cached_dataset "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathgen_vqa_311k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathgen_cotadd_12k" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathmmu_6k_hypocritical" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathvqa_12k_hypocritical" "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_classification_100k_hypocritical" \
    --num_train_epochs 1 \
    --split_dataset_ratio 0.05 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 18 \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 1 \
    --max_length 3072 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --dataset_num_proc 6 \
    --save_total_limit 5 \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attention_2 \
    --check_model false \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --lora_rank ${LORA_RANK} \
    --lora_alpha $((LORA_RANK * 2)) \
    --target_modules all-linear \
    --early_stop_interval 3 \
    --metric_for_best_model loss \
    --load_best_model_at_end true \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_sft_kdw${KD_LOSS_WEIGHT}_lorarank${LORA_RANK}_hypocritical \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_sft_kdw${KD_LOSS_WEIGHT}_lorarank${LORA_RANK}_hypocritical/logs \
    --kd_teacher_model_type conchv1_5 uni2 virchow2 \
    --kd_teacher_model_path /data/ckpt/conchv1.5/pytorch_model_vision.bin /data/ckpt/uni2/pytorch_model.bin /data/ckpt/virchow2/pytorch_model.bin \
    --kd_loss_weight ${KD_LOSS_WEIGHT} \
    --resume_from_checkpoint /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_sft_kdw0.5_lorarank16_hypocritical/v0-20251222-194902/checkpoint-200-merged/

# 在单一数据集训练 试验
# 直接sft
export LORA_RANK=16
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29510 \
CUDA_VISIBLE_DEVICES=2,3 \
swift sft \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --train_type lora \
    --cached_dataset "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathvqa_12k" \
    --num_train_epochs 1 \
    --split_dataset_ratio 0.05 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 18 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 1 \
    --max_length 3072 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --dataset_num_proc 6 \
    --save_total_limit 5 \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attention_2 \
    --check_model false \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --lora_rank ${LORA_RANK} \
    --lora_alpha $((LORA_RANK * 2)) \
    --target_modules all-linear \
    --early_stop_interval 3 \
    --metric_for_best_model loss \
    --load_best_model_at_end true \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft_vqa/qwen3_vl_4b_sft_lorarank${LORA_RANK} \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft_vqa/qwen3_vl_4b_sft_lorarank${LORA_RANK}/logs \
    


# 直接sft with kd
# Todo: 先执行11小时睡眠再执行下面命令
sleep 9h

export KD_LOSS_WEIGHT=0.9
export LORA_RANK=16
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MASTER_PORT=29519 \
CUDA_VISIBLE_DEVICES=4,5 \
swift sft \
    --model /data/ckpt/Qwen3-VL-4B-Instruct \
    --train_type lora \
    --cached_dataset "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cache_pathvqa_12k" \
    --num_train_epochs 1 \
    --split_dataset_ratio 0.05 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 18 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 1 \
    --max_length 3072 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 6 \
    --dataset_num_proc 6 \
    --save_total_limit 5 \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attention_2 \
    --check_model false \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --lora_rank ${LORA_RANK} \
    --lora_alpha $((LORA_RANK * 2)) \
    --target_modules all-linear \
    --early_stop_interval 3 \
    --metric_for_best_model loss \
    --load_best_model_at_end true \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft_vqa/qwen3_vl_4b_sft_kdw${KD_LOSS_WEIGHT}_lorarank${LORA_RANK} \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output4paper/sft_vqa/qwen3_vl_4b_sft_kdw${KD_LOSS_WEIGHT}_lorarank${LORA_RANK}/logs \
    --kd_teacher_model_type conchv1_5 uni2 virchow2 \
    --kd_teacher_model_path /data/ckpt/conchv1.5/pytorch_model_vision.bin /data/ckpt/uni2/pytorch_model.bin /data/ckpt/virchow2/pytorch_model.bin \
    --kd_loss_weight ${KD_LOSS_WEIGHT}
