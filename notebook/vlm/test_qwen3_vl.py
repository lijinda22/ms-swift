from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer
from threading import Thread
import torch

# model = AutoModelForImageTextToText.from_pretrained(
#     "Qwen/Qwen3-VL-235B-A22B-Instruct", dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.

# model_name = "/data/ckpt/Patho-R1-7B/"
# model_name = "/data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_cpt_sft_kdw0.5_lorarank16_hypocritical/v0-20251227-160912/checkpoint-2922-merged/"
# model_name = "/data/ljd/Pathology_FM_LLM/expriment/output4paper/grpo/qwen3_vl_4b_cpt_sft_kd_mmu_lr5e-6/v0-20251230-144744/checkpoint-1200-merged/"
model_name = "/data/ckpt/Qwen3-VL-4B-Thinking"
# model_name = "/data/ckpt/Qwen3-VL-4B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_name)

CLOSE_QUESTION_TEMPLATE = "{Question}\nPlease output only the final answer option directly. Just one letter (A, B, C, or D) with no explanation or additional text."
COT_QUESTION_TEMPLATE = "{Question}\nThink through the question step by step, enclose your reasoning process in <think>...</think> tags. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags. No extra information or text outside of these tags."
query = "What is the nature of the stroma in the image?\nA) Eosinophilic\nB) Hyalinized\nC) Edematous\nD) Myxoid"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/data/dataset/vqa/PathMMU/images/70959f1c002947d00f19fa662bf0dde81b14cc4ac67a82e5534cfa776dd8b1ff.png",
            },
            {
                "type": "text",
                # "text": CLOSE_QUESTION_TEMPLATE.format(Question=query),
                "text": COT_QUESTION_TEMPLATE.format(Question=query),
            },
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=2048)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for new_text in streamer:
    print(new_text, end="", flush=True)
print()
