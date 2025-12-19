import os
import json
import random
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
from tqdm import tqdm
import math
from qwen_vl_utils import process_vision_info
import numpy as np

# Configuration
MODEL_PATH = "/data/ckpt/Lingshu-32B"
TENSOR_PARALLEL_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 1
GPU_MEMORY_UTILIZATION = 0.95
MAX_IMAGES_PER_PROMPT = 1

INPUT_FILE = "/data/ljd/VLM-R1/dataset/sft/pathgen_instruct_close_subset_96289.jsonl"
OUTPUT_FILE = "/data/ljd/VLM-R1/dataset/sft/pathgen_instruct_close_cot_50k_new.jsonl"

BATCH_SIZE = 8
SAMPLING_TEMP_COT = 0.2
SAMPLING_TOP_P_COT = 0.9
MAX_TOKENS_COT = 512
SAMPLE_COUNT = 50000

print("Initializing processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

def initialize_llm():
    print(f"Initializing LLM with TP={TENSOR_PARALLEL_SIZE}...")
    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": MAX_IMAGES_PER_PROMPT},
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    print("Model and processor initialized.")
    return llm

def generate_cot_data(data_to_process, output_filepath, batch_size, llm):
    print(f"\n--- Starting CoT Data Generation ---")
    print(f"Processing {len(data_to_process)} samples for CoT generation.")
    print(f"Output file: {output_filepath}")

    # Resume capability: Use (image_path, question) as a unique key
    processed_keys = set()
    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_keys.add((data["image_path"], data["question"]))
                except:
                    continue
        
        if processed_keys:
            data_to_process = [
                item for item in data_to_process 
                if (item.get("image"), item.get("question")) not in processed_keys
            ]
            print(f"Resuming: {len(processed_keys)} (image, question) pairs already processed. Remaining items to process: {len(data_to_process)}")

    sampling_params = SamplingParams(
        temperature=SAMPLING_TEMP_COT,
        top_p=SAMPLING_TOP_P_COT,
        max_tokens=MAX_TOKENS_COT,
    )

    with open(output_filepath, "a", encoding="utf-8") as outfile:
        for i in tqdm(range(0, len(data_to_process), batch_size)):
            batch_data = data_to_process[i : i + batch_size]
            batch_prompts = []
            batch_mm_data = []
            original_data_batch = []

            for idx, item in enumerate(batch_data):
                image_path = item.get("image")
                question_text = item.get("question")
                correct_answer = item.get("answer")
                
                if not all([image_path, question_text, correct_answer]):
                    continue
                if not os.path.exists(image_path):
                    continue

                try:
                    abs_image_path = os.path.abspath(image_path)
                    image = Image.open(abs_image_path).convert("RGB")
                except:
                    continue

                prompt_text = (
                    f"You are a medical expert. Your task is to generate a chain-of-thought reasoning for a given medical image and question. "
                    f"The reasoning should be a concise, single paragraph that explains the logical steps to arrive at the correct answer, focusing on visual evidence in the image. "
                    f"Do not use a numbered or bulleted list. The output should only contain your thinking process.\n\n"
                    f"Question: {question_text}\n"
                    f"Correct Answer: {correct_answer}\n\n"
                    f"Reasoning:"
                )

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]

                prompt_string = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, _ = process_vision_info(messages)
                mm_data_item = {"image": image_inputs}

                batch_prompts.append(prompt_string)
                batch_mm_data.append(mm_data_item)
                original_data_batch.append(item)

            if not batch_prompts:
                continue

            processed_inputs = [
                {"prompt": p, "multi_modal_data": mm}
                for p, mm in zip(batch_prompts, batch_mm_data)
            ]

            try:
                outputs = llm.generate(processed_inputs, sampling_params, use_tqdm=False)
                for output, original_item in zip(outputs, original_data_batch):
                    generated_cot = output.outputs[0].text.strip()
                    if generated_cot.startswith("<think>"):
                        generated_cot = generated_cot[len("<think>") :]
                    if generated_cot.endswith("</think>"):
                        generated_cot = generated_cot[: -len("</think>")]
                    generated_cot = generated_cot.strip()

                    result = {
                        "image_path": original_item.get("image"),
                        "question": original_item.get("question"),
                        "correct_answer": original_item.get("answer"),
                        "slide_id": original_item.get("slide_id"),
                        "reasoning_cot_lingshu32b": generated_cot,
                        "request_id": output.request_id,
                    }
                    outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                outfile.flush()
            except Exception as e:
                print(f"Error in batch {i}: {e}")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"Loading data from {INPUT_FILE}...")
    all_data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
    
    print(f"Loaded {len(all_data)} samples.")
    
    if len(all_data) > SAMPLE_COUNT:
        sampled_data = random.sample(all_data, SAMPLE_COUNT)
        print(f"Sampled {SAMPLE_COUNT} items.")
    else:
        sampled_data = all_data
        print(f"Data size {len(all_data)} is less than or equal to {SAMPLE_COUNT}, using all data.")

    llm = initialize_llm()
    generate_cot_data(sampled_data, OUTPUT_FILE, BATCH_SIZE, llm)
    print("\nCoT data generation completed.")
