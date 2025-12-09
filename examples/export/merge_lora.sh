# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
swift export \
    --adapters /data/ljd/Pathology_FM_LLM/expriment/output/qwen3_vl_2b_cpt_sft_kd/v0-20251204-105138/checkpoint-2628/ \
    --merge_lora true
