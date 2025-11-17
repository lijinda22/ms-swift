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

MODEL_PATH = "/data/ckpt/Lingshu-7B"
TENSOR_PARALLEL_SIZE = 2
GPU_MEMORY_UTILIZATION = 0.9
MAX_IMAGES_PER_PROMPT = 4

DETAIL_DESC_INPUT_FILE = "/data/ljd/VLM-R1/dataset/pretrain/pathgen_pair_1586502.json"
DETAIL_DESC_OUTPUT_FILE = "/data/ljd/VLM-R1/dataset/sft/pathgen_detail_desc.jsonl"

BATCH_SIZE = 32
SAMPLING_TEMP_DETAIL = 0.7
SAMPLING_TOP_P_DETAIL = 0.9
MAX_TOKENS_DETAIL = 1024

IMAGES_PER_SLIDE_ID = 10

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


def get_processed_indices(output_filepath):
    processed_indices = set()
    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "request_id" in data:
                            processed_indices.add(data["request_id"])
                    except:
                        pass
    return processed_indices


def generate_detailed_descriptions(input_filepath, output_filepath, batch_size, llm):
    print(f"\n--- Starting Detailed Description Generation ---")
    print(f"Input file: {input_filepath}")
    print(f"Output file: {output_filepath}")

    data_to_process = []
    print("Loading data for detailed descriptions...")
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        print(f"Loaded {len(all_data)} total samples.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    slide_groups = {}
    for item in all_data:
        slide_id = item.get("slide_id")
        if slide_id not in slide_groups:
            slide_groups[slide_id] = []
        slide_groups[slide_id].append(item)

    for slide_id, items in slide_groups.items():
        if len(items) > IMAGES_PER_SLIDE_ID:
            selected_items = random.sample(items, IMAGES_PER_SLIDE_ID)
            data_to_process.extend(selected_items)
        else:
            data_to_process.extend(items)

    print(
        f"Selected {len(data_to_process)} samples for processing ({IMAGES_PER_SLIDE_ID} images per slide_id)."
    )

    sampling_params = SamplingParams(
        temperature=SAMPLING_TEMP_DETAIL,
        top_p=SAMPLING_TOP_P_DETAIL,
        max_tokens=MAX_TOKENS_DETAIL,
    )

    results = []
    print(f"Starting inference with batch size {batch_size}...")

    start_index = 0
    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                start_index = len(lines) * batch_size
                print(f"Resuming from index {start_index}")

    with open(output_filepath, "a", encoding="utf-8") as outfile:
        for i in tqdm(range(start_index, len(data_to_process), batch_size)):
            batch_data = data_to_process[i : i + batch_size]
            batch_prompts = []
            batch_mm_data = []

            valid_batch_indices = []
            original_data_batch = []

            for idx, item in enumerate(batch_data):
                image_path = item.get("img")
                original_caption = item.get("text", "")

                if not image_path or not os.path.exists(image_path):
                    print(
                        f"Warning: Image path not found or invalid for sample {i+idx}: {image_path}. Skipping."
                    )
                    continue

                try:
                    abs_image_path = os.path.abspath(image_path)
                    image = Image.open(abs_image_path).convert("RGB")
                except Exception as e:
                    print(
                        f"Warning: Could not open image {abs_image_path} for sample {i+idx}: {e}. Skipping."
                    )
                    continue

                prompt_text = (
                    f"Please provide a detailed description of the provided histopathology image. "
                    f"Here is a previous, potentially less detailed caption for reference: '{original_caption}'. "
                    f"Focus on describing the visual features in detail, expanding on or correcting the reference caption based *only* on what you see in the image. "
                    f"Describe cellular structures, tissue organization, morphological patterns, staining characteristics, and any abnormalities observed."
                )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]

                prompt_string = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                mm_data_item = {"image": image_inputs}

                batch_prompts.append(prompt_string)
                batch_mm_data.append(mm_data_item)
                valid_batch_indices.append(i + idx)
                original_data_batch.append(item)

            if not batch_prompts:
                print(f"Warning: Empty batch at index {i}. Skipping.")
                continue

            processed_inputs = [
                {"prompt": p, "multi_modal_data": mm}
                for p, mm in zip(batch_prompts, batch_mm_data)
            ]

            try:
                outputs = llm.generate(
                    processed_inputs, sampling_params, use_tqdm=False
                )
            except Exception as e:
                print(
                    f"Error during vLLM generation for batch starting at index {i}: {e}"
                )
                continue

            for output, original_item in zip(outputs, original_data_batch):
                generated_text = output.outputs[0].text.strip()
                result = {
                    "image_path": original_item.get("img"),
                    "original_caption": original_item.get("text"),
                    "slide_id": original_item.get("slide_id"),
                    "detailed_description_lingshu32b": generated_text,
                    "request_id": output.request_id,
                }
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
            outfile.flush()

    print(f"--- Detailed Description Generation Finished ---")
    print(f"Results saved to {output_filepath}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    llm = initialize_llm()

    generate_detailed_descriptions(
        DETAIL_DESC_INPUT_FILE, DETAIL_DESC_OUTPUT_FILE, BATCH_SIZE, llm
    )

    print("\nDetailed description generation completed.")
