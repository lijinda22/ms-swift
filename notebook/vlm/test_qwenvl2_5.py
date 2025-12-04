from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# 3个都是 Qwen2.5-VL 模型, 经过不同微调训练得到的, 原生模型和微调的领域模型
# /data/ckpt/Qwen2.5-VL-7B-Instruct/
# /data/ckpt/Patho-R1-7B/
# /data/ckpt/Lingshu-7B/
# /data/ckpt/Lingshu-32B/
model_path = "/data/ckpt/Patho-R1-7B"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)

# example question from Pathmmu-test-dataset
# ground truth: D
# Reasoning style options (choose one):
# - Chain-of-Draft, a concise reasoning prompting strategy (COD):
# You are a pathology expert, your task is to think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return the answer at the end of the response after a separator. Use the following format:<think> Your step-by-step reasoning </think><answer> Your final answer </answer>
# - Chain-of-Thought (COT):
messages = [
    {   
        "role": "system",
        "content": "You are a pathology expert, your task is to answer question step by step. Use the following format:<think> Your step-by-step reasoning </think><answer> Your final answer </answer>"
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/data/dataset/classification/CCRCC/tissue_classification/cancer/TCGA-B0-5691_27828_20422_cancer_11339.png",
            },
            {
                "type": "text", 
                "text": "Which of the following classes does this pathology image belong to: \"blood\", \"cancer\", \"normal\", \"stroma\"?\nA. Red blood cells\nB. Renal cancer\nC. Normal renal\nD. Stromal, including smooth muscle, fibrous stroma and blood vessels"
            },       
        ],
    }
]
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=2048)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)