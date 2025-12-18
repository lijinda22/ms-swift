from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer
from threading import Thread
import torch

# model = AutoModelForImageTextToText.from_pretrained(
#     "Qwen/Qwen3-VL-235B-A22B-Instruct", dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.

model_name = "/data/ckpt/Qwen3-VL-4B-Thinking"
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

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/data/ljd/VLM-R1/ckpt/vlm-r1-rec-dataset/train2014/COCO_train2014_000000556824.jpg",
            },
            {
                "type": "text",
                "text": "图中人在干什么?",
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
