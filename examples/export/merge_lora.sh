# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
swift export \
    --adapters /data/ljd/Pathology_FM_LLM/expriment/output4paper/qwen3_vl_2b_sft_kd_0.5/v0-20251213-135528/checkpoint-800/ \
    --merge_lora true
