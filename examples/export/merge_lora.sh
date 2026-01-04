# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
CUDA_LAUNCH_BLOCKING=1 swift export \
    --adapters /data/ljd/Pathology_FM_LLM/expriment/output4paper/grpo/qwen3_vl_4b_cpt_sft_kd_mmu_lr5e-6/v0-20251230-144744/checkpoint-1200/ \
    --merge_lora true \
    --device_map cpu
