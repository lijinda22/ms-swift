from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer
from threading import Thread
import torch

# model = AutoModelForImageTextToText.from_pretrained(
#     "Qwen/Qwen3-VL-235B-A22B-Instruct", dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.

model_name = "/data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_sft_kdw0.5_lorarank16_hypocritical/v4-20251224-173742/checkpoint-2922-merged/"
# model_name = "/data/ckpt/Qwen3-VL-4B-Thinking"
# model_name = "/data/ckpt/Qwen3-VL-4B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_name)
# processor = AutoProcessor.from_pretrained("/data/ckpt/Qwen3-VL-2B-Thinking")
# processor = AutoProcessor.from_pretrained("/data/ckpt/Qwen3-VL-2B-Instruct")

CLOSE_QUESTION_TEMPLATE = "{Question}\nPlease output only the final answer option directly. Just one letter (A, B, C, or D) with no explanation or additional text."
COT_QUESTION_TEMPLATE = "{Question}\nThink through the question step by step, enclose your reasoning process in <think>...</think> tags. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags. No extra information or text outside of these tags."
query = "In the top left image (a), what does the arrow specifically indicate within the histological features present?\nA) Fibrin deposition\nB) Neutrophil infiltration\nC) Eosinophilic cytoplasmic staining\nD) Synovial lining hyperplasia"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/data/dataset/vqa/PathMMU/images/af6475ca3b67f195223f863a3a895fd1cc7dc5f1919e927eb483645f32ae414e.png",
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
generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=512)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for new_text in streamer:
    print(new_text, end="", flush=True)
print()
