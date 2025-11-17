# If not using flash_attn, or transformers<4.44,
# or encountering an abnormally large loss (i.e., the model does not support packing),
# please remove `--packing true`.

nproc_per_node=4 \
NPROC_PER_NODE=$nproc_per_node \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=2,3,4,5 \
swift pt \
    --model /data/ckpt/Qwen3-VL-2B-Instruct \
    --train_type full \
    --dataset "/data/ljd/VLM-R1/dataset/pretrain/pretrain_dataset.jsonl" \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --split_dataset_ratio 0.05 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --packing true \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --deepspeed zero3 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --padding_free true \
    --save_only_model true \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_pretrain \
    --attn_impl flash_attn \
    --check_model false \
    --load_from_cache_file true \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_pretrain/logs \

    
    # --streaming true \
    # --max_steps 10000 \


nproc_per_node=2 \
NPROC_PER_NODE=$nproc_per_node \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_PORT=29502 \
swift pt \
    --model /data/ckpt/Qwen3-VL-2B-Instruct \
    --train_type full \
    --dataset "/data/ljd/VLM-R1/dataset/pretrain/pretrain_dataset.jsonl" \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --split_dataset_ratio 0.99 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --packing true \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --deepspeed zero3 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --padding_free true \
    --save_only_model true \
    --output_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_pretrain_test \
    --attn_impl flash_attn \
    --check_model false \
    --load_from_cache_file true \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --report_to tensorboard \
    --logging_dir /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_pretrain_test/logs \
