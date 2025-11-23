# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
swift export \
    --adapters /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_sft/v1-20251122-120152/checkpoint-1246 \
    --merge_lora true
