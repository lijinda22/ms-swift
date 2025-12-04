import os
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np

# --- Configuration ---
DATA_DIR = "/data/dataset/vqa/path-vqa/data"
REFINE_DIR = "/data/dataset/vqa/path-vqa/data_refine"
MODEL_PATH = "/data/ckpt/Lingshu-7B"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.95
MAX_IMAGES_PER_PROMPT = 1
BATCH_SIZE = 16

import shutil

# Dataset split mapping to folder names
SPLIT_MAPPING = {
    "train": "train",
    "validation": "eval",
    "test": "test"
}

def save_images_and_metadata(dataset, split_key):
    folder_name = SPLIT_MAPPING.get(split_key, split_key)
    # Save initial images to a 'raw' folder to avoid confusion with classified folders later
    split_dir = os.path.join(REFINE_DIR, folder_name, "raw")
    os.makedirs(split_dir, exist_ok=True)
    
    metadata_list = []
    json_path = os.path.join(REFINE_DIR, f"pathvqa_{folder_name}_raw.json")
    
    if os.path.exists(json_path):
        print(f"Metadata for {split_key} (raw) already exists at {json_path}. Loading...")
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f), folder_name

    print(f"Processing {split_key} split -> {folder_name}...")
    
    data_split = dataset[split_key]
    
    for idx, item in enumerate(tqdm(data_split, desc=f"Saving {folder_name} images")):
        image = item['image']
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image_filename = f"{idx}.jpg"
        image_path = os.path.join(split_dir, image_filename)
        
        # Save image
        if not os.path.exists(image_path):
            image.save(image_path)
            
        metadata_list.append({
            "image": image_path,
            "question": item['question'],
            "answer": item['answer'],
            "split": folder_name,
            "original_idx": idx
        })
        
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        
    return metadata_list, folder_name

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
    return llm

def filter_and_move_images(llm, processor, data, folder_name):
    base_split_dir = os.path.join(REFINE_DIR, folder_name)
    pathology_dir = os.path.join(base_split_dir, "pathology")
    non_pathology_dir = os.path.join(base_split_dir, "non_pathology")
    
    os.makedirs(pathology_dir, exist_ok=True)
    os.makedirs(non_pathology_dir, exist_ok=True)

    pathology_json_path = os.path.join(REFINE_DIR, f"pathvqa_{folder_name}_pathology.json")
    non_pathology_json_path = os.path.join(REFINE_DIR, f"pathvqa_{folder_name}_filtered.json")

    if os.path.exists(pathology_json_path) and os.path.exists(non_pathology_json_path):
        print(f"Filtered output for {folder_name} already exists. Skipping.")
        return

    print(f"Filtering and moving {len(data)} images for {folder_name}...")
    
    prompt_text = (
        "You are a medical expert. Look at this image. "
        "Is it a microscopic histopathological image of tissue? "
        "Please answer with only 'Yes' or 'No'."
    )
    
    kept_data = []
    filtered_out_data = []
    
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=10)
    
    # Batch processing
    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Filtering"):
        batch = data[i:i+BATCH_SIZE]
        batch_inputs = []
        valid_indices = []
        
        for idx, item in enumerate(batch):
            image_path = item['image']
            try:
                # Load image for inference
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error opening {image_path}: {e}")
                continue
                
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            batch_inputs.append({
                "prompt": text,
                "multi_modal_data": {"image": image_inputs}
            })
            valid_indices.append(idx)
            
        if not batch_inputs:
            continue
            
        outputs = llm.generate(batch_inputs, sampling_params, use_tqdm=False)
        
        for j, output in enumerate(outputs):
            response = output.outputs[0].text.strip().lower()
            original_item = batch[valid_indices[j]]
            old_path = original_item['image']
            filename = os.path.basename(old_path)
            
            # Determine destination
            if "yes" in response:
                new_dir = pathology_dir
                is_pathology = True
            else:
                new_dir = non_pathology_dir
                is_pathology = False
            
            new_path = os.path.join(new_dir, filename)
            
            # Move file
            # We check if src exists because we might have moved it in a previous partial run
            # Or if output exists, we skip moving
            if os.path.exists(old_path) and not os.path.exists(new_path):
                shutil.move(old_path, new_path)
            elif os.path.exists(new_path):
                # Already moved
                pass
            else:
                print(f"Warning: Source file missing {old_path}")
            
            # Update item
            original_item['image'] = new_path
            original_item['is_pathological'] = is_pathology
            
            if is_pathology:
                kept_data.append(original_item)
            else:
                filtered_out_data.append(original_item)
            
    print(f"Split {folder_name}: Kept {len(kept_data)} (Pathology), Filtered {len(filtered_out_data)} (Non-Pathology).")
    
    with open(pathology_json_path, 'w', encoding='utf-8') as f:
        json.dump(kept_data, f, indent=2, ensure_ascii=False)
        
    with open(non_pathology_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_out_data, f, indent=2, ensure_ascii=False)

def main():
    # 1. Load Dataset
    print(f"Loading dataset from {DATA_DIR}...")
    try:
        dataset = load_dataset("parquet", data_dir=DATA_DIR)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Prepare Metadata & Images
    all_metadata_info = [] # List of (metadata, folder_name)
    for split_key in ['train', 'validation', 'test']:
        if split_key in dataset:
            meta, folder_name = save_images_and_metadata(dataset, split_key)
            all_metadata_info.append((meta, folder_name))

    # 3. Initialize Model
    print("Initializing model for filtering...")
    try:
        llm = initialize_llm()
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # 4. Run Filter and Move
    for data, folder_name in all_metadata_info:
        filter_and_move_images(llm, processor, data, folder_name)
    
    print("\nPipeline completed.")

if __name__ == "__main__":
    main()
