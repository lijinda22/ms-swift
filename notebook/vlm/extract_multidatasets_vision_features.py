import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import json
import os
from tqdm import tqdm
import argparse
import gc
from os.path import join as j_

# Configuration
MODEL_PATHS = {
    "qwen3_vl-4b-instruct": "/data/ckpt/Qwen3-VL-4B-Instruct/",
    # "qwen3_vl-4b-thinking": "/data/ckpt/Qwen3-VL-4B-Thinking/",
    "lingshu-7b": "/data/ckpt/Lingshu-7B/",
    "qwen3_vl-4b-sft": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_sft_lorarank16/v1-20251222-215426/checkpoint-2922-merged/",
    "qwen3_vl-4b-sft-kd-w0.5_hypocritical": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_sft_kdw0.5_lorarank16_hypocritical/v4-20251224-173742/checkpoint-2922-merged/",
    "qwen3_vl-4b-cpt-sft": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_cpt_sft_lorarank16/v0-20251225-042643/checkpoint-2922-merged/",
    "qwen3_vl-4b-cpt-sft-kd-w0.5_hypocritical": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_cpt_sft_kdw0.5_lorarank16_hypocritical/v0-20251227-160912/checkpoint-2922-merged/",
}

DATASETS_BASE_DIR = "/data/ljd/Pathology_FM_LLM/expriment/classify"
DATASETS_CONFIG = [
    {"name": "CCRCC", "path": j_(DATASETS_BASE_DIR, "CCRCC")},
    {"name": "BreaKHis", "path": j_(DATASETS_BASE_DIR, "BreaKHis")},
    {"name": "chaoyang", "path": j_(DATASETS_BASE_DIR, "chaoyang")},
    {"name": "crc100k", "path": j_(DATASETS_BASE_DIR, "crc100k")},
    {"name": "CRC_MSI", "path": j_(DATASETS_BASE_DIR, "CRC_MSI")},
    {"name": "PanCancer-TIL", "path": j_(DATASETS_BASE_DIR, "PanCancer-TIL")},
]

def load_model(model_key):
    model_path = MODEL_PATHS[model_key]
    print(f"Loading model {model_key} from {model_path}...")
    
    if "qwen3" in model_key:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True
        )
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

def extract_features(model, processor, data_batch, model_key):
    images = []
    messages_batch = []
    
    for item in data_batch:
        image_path = item['image_path']
        try:
            image = Image.open(image_path).convert("RGB")
            images.append(image)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
            messages_batch.append(messages)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    if not messages_batch:
        return None

    # Prepare inputs
    if "qwen3" in model_key:
        text = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        inputs = processor(
            text=text,
            images=images,
            padding=True,
            return_tensors="pt",
        )
    else:
        text = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )

    inputs = inputs.to(model.device)
    
    # Extract features
    visual_model = model.model.visual
    pixel_values = inputs.pixel_values
    image_grid_thw = inputs.image_grid_thw
    
    with torch.no_grad():
        pixel_values = pixel_values.type(visual_model.dtype)
        image_embeds = visual_model(pixel_values, grid_thw=image_grid_thw)
        if isinstance(image_embeds, tuple):
            image_embeds = image_embeds[0]

    # Split features back to per-image
    merge_size = visual_model.spatial_merge_size
    split_sizes = (torch.prod(image_grid_thw, dim=1) // (merge_size**2)).tolist()
    
    total_tokens = image_embeds.shape[0]
    expected_tokens = sum(split_sizes)
    
    if total_tokens == expected_tokens:
        features = list(torch.split(image_embeds, split_sizes))
    elif total_tokens == sum([s + 1 for s in split_sizes]):
        split_sizes = [s + 1 for s in split_sizes]
        features = list(torch.split(image_embeds, split_sizes))
    else:
        print(f"Warning: Token count mismatch. Total: {total_tokens}, Expected: {expected_tokens}")
        return None

    pooled_features = []
    for feat in features:
        pooled = feat.mean(dim=0) # GAP
        pooled_features.append(pooled.cpu())
        
    return pooled_features

def process_dataset_split(dataset_name, dataset_root, split, model, processor, model_key):
    json_path = j_(dataset_root, f"{split}.json")
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    print(f"Processing {dataset_name} - {split}...")
    output_root = j_(dataset_root, "vlm_vision_only")
    feat_dir = j_(output_root, split, "feat")
    label_dir = j_(output_root, split, "label")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    feat_save_path = j_(feat_dir, f"{model_key}.pt")
    label_save_path = j_(label_dir, f"{model_key}.json")
    if os.path.exists(feat_save_path) and os.path.exists(label_save_path):
        print(f"Features and labels already exist for {dataset_name} - {split}. Skipping.")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_features = []
    all_labels = []
    
    batch_size = 16 
    
    for i in tqdm(range(0, len(data), batch_size)):
        batch_items = data[i : i + batch_size]
        
        # Extract labels
        batch_labels = []
        for item in batch_items:
            if 'gt_label' in item:
                batch_labels.append(item['gt_label'])
            else:
                batch_labels.append("unknown")
        
        # Extract features
        features = extract_features(model, processor, batch_items, model_key)
        
        if features:
            all_features.extend(features)
            all_labels.extend(batch_labels)
        else:
            print(f"Skipping batch {i}")
            
    # Save results
    torch.save(torch.stack(all_features), feat_save_path)
    print(f"Saved features to {feat_save_path}")
    
    with open(label_save_path, 'w') as f:
        json.dump(all_labels, f)
    print(f"Saved labels to {label_save_path}")

def main():
    for model_key in MODEL_PATHS.keys():
        print(f"\n\n{'='*30}\nLoading Model: {model_key}\n{'='*30}")
        try:
            model, processor = load_model(model_key)
            model.eval()
            
            for ds_conf in DATASETS_CONFIG:
                print(f"\n--- Dataset: {ds_conf['name']} ---")
                process_dataset_split(ds_conf["name"], ds_conf["path"], "train", model, processor, model_key)
                process_dataset_split(ds_conf["name"], ds_conf["path"], "test", model, processor, model_key)
                
            del model
            del processor
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing model {model_key}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
