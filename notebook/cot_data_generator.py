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

MODEL_PATH = "/data/ckpt/Lingshu-32B"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.92
MAX_IMAGES_PER_PROMPT = 1

MCQ_INPUT_FILE = "/data/ljd/VLM-R1/dataset/sft/pathgen_instruct_close_137555.json"

BATCH_SIZE = 36
SAMPLING_TEMP_COT = 0.2
SAMPLING_TOP_P_COT = 0.9
MAX_TOKENS_COT = 512

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


def generate_cot_data(data_to_process, output_filepath, batch_size, llm):
    print(f"\n--- Starting CoT Data Generation ---")
    print(f"Processing up to {len(data_to_process)} samples for CoT generation.")
    print(f"Output file: {output_filepath}")

    # --- IMPROVEMENT 1: Robust resumption ---
    # Check for already processed samples in the output file to avoid duplicates
    if os.path.exists(output_filepath):
        processed_image_paths = set()
        with open(output_filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    # Use image_path as a unique identifier for processed samples
                    processed_image_paths.add(json.loads(line)["image_path"])
                except (json.JSONDecodeError, KeyError):
                    # Skip malformed lines in the output file
                    continue

        if processed_image_paths:
            original_count = len(data_to_process)
            # Filter out samples that have already been processed
            data_to_process = [
                item
                for item in data_to_process
                if item.get("image") not in processed_image_paths
            ]
            print(
                f"Resuming generation: {len(processed_image_paths)} samples already found in output file."
            )
            print(
                f"Filtered out processed samples. Remaining samples to generate: {len(data_to_process)}"
            )

    sampling_params = SamplingParams(
        temperature=SAMPLING_TEMP_COT,
        top_p=SAMPLING_TOP_P_COT,
        max_tokens=MAX_TOKENS_COT,
    )

    print(f"Starting inference with batch_size {batch_size}...")

    with open(output_filepath, "a", encoding="utf-8") as outfile:
        # Loop from the beginning of the (potentially filtered) data
        for i in tqdm(range(0, len(data_to_process), batch_size)):
            batch_data = data_to_process[i : i + batch_size]
            batch_prompts = []
            batch_mm_data = []
            valid_batch_indices = []
            original_data_batch = []

            for idx, item in enumerate(batch_data):
                image_path = item.get("image")
                question_text = item.get("question")
                correct_answer = item.get("answer")
                slide_id = item.get("slide_id")

                if not all([image_path, question_text, correct_answer]):
                    print(f"Warning: Missing data in MCQ sample {i+idx}. Skipping.")
                    continue

                if not os.path.exists(image_path):
                    print(
                        f"Warning: Image path not found for MCQ sample {i+idx}: {image_path}. Skipping."
                    )
                    continue

                try:
                    abs_image_path = os.path.abspath(image_path)
                    image = Image.open(abs_image_path).convert("RGB")
                except Exception as e:
                    print(
                        f"Warning: Could not open image {abs_image_path} for MCQ sample {i+idx}: {e}. Skipping."
                    )
                    continue

                prompt_text = (
                    f"You are a medical expert. Your task is to generate a chain-of-thought reasoning for a given medical image and question. "
                    f"The reasoning should be a concise, single paragraph that explains the logical steps to arrive at the correct answer, focusing on visual evidence in the image. "
                    f"Do not use a numbered or bulleted list. The output should only contain your thinking process.\n\n"
                    f"Here is an example of a good reasoning process:\n"
                    f"The image presented is a transverse CT scan of the abdomen and pelvis. The presence of calculi (urines filled with stones or grit) in the pelvic organs is a consistent finding in urolithiasis.\n\n"
                    f"Now, generate the reasoning for the following:\n"
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
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, video_inputs = process_vision_info(messages)
                mm_data_item = {"image": image_inputs}

                batch_prompts.append(prompt_string)
                batch_mm_data.append(mm_data_item)
                valid_batch_indices.append(i + idx)
                original_data_batch.append(item)

            if not batch_prompts:
                print(f"Warning: Empty batch at index {i} for CoT. Skipping.")
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
                    f"Error during vLLM generation for CoT batch starting at index {i}: {e}"
                )
                continue

            for output, original_item in zip(outputs, original_data_batch):
                generated_cot = output.outputs[0].text.strip()

                # --- IMPROVEMENT 2: Clean LLM output ---
                # Remove <think> tags if the model includes them
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

    print(f"--- CoT Data Generation Finished ---")
    print(f"Results saved to {output_filepath}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    COT_OUTPUT_FILE = "/data/ljd/VLM-R1/dataset/sft/pathgen_instruct_close_cot.jsonl"
    SUBSET_OUTPUT_FILE = (
        "/data/ljd/VLM-R1/dataset/sft/pathgen_instruct_close_subset.jsonl"
    )

    print(f"Loading data from {MCQ_INPUT_FILE}...")
    try:
        with open(MCQ_INPUT_FILE, "r", encoding="utf-8") as f:
            all_mcq_data = json.load(f)
        print(f"Loaded {len(all_mcq_data)} total samples.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {MCQ_INPUT_FILE}")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    random.shuffle(all_mcq_data)
    split_index = int(len(all_mcq_data) * 0.3)
    cot_data_to_process = all_mcq_data[:split_index]
    subset_data_to_save = all_mcq_data[split_index:]

    print(
        f"Splitting data: {len(cot_data_to_process)} for CoT, {len(subset_data_to_save)} for subset."
    )

    print(f"Saving {len(subset_data_to_save)} samples to {SUBSET_OUTPUT_FILE}...")
    with open(SUBSET_OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in tqdm(subset_data_to_save, desc="Saving subset"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Subset data saved.")

    llm = initialize_llm()

    generate_cot_data(cot_data_to_process, COT_OUTPUT_FILE, BATCH_SIZE, llm)

    print("\nCoT data generation completed.")
