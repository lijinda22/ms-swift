import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import json
import os
from tqdm import tqdm
import argparse
import gc

# Configuration
MODEL_PATHS = {
    "qwen2.5_vl-7b": "/data/ckpt/Qwen2.5-VL-7B-Instruct/",
    "patho-r1-7b": "/data/ckpt/Patho-R1-7B/",
    "lingshu-7b": "/data/ckpt/Lingshu-7B/",
    # "lingshu-32b": "/data/ckpt/Lingshu-32B/",
    "qwen3_vl-2b": "/data/ckpt/Qwen3-VL-2B-Instruct/",
}

DATA_ROOT = "/data/ljd/Pathology_FM_LLM/expriment/classify/CCRCC/"
OUTPUT_ROOT = "/data/ljd/Pathology_FM_LLM/expriment/classify/CCRCC/vlm_vision_only"

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

def get_label_from_path(image_path):
    # Path format: .../class_name/image_name
    # or .../class_name/UPPER_EXT/image_name
    # Based on divide_ccrcc_dataset.py:
    # dataroot = "/data/dataset/classification/CCRCC/tissue_classification/"
    # cls_dir = pjoin(dataroot, cls_name)
    # images are in cls_dir
    
    # So the parent directory name should be the label.
    # Let's verify against the known classes.
    classes = ["blood", "cancer", "normal", "stroma"]
    
    parts = image_path.split(os.sep)
    # Check parent and grandparent
    if parts[-2] in classes:
        return parts[-2]
    elif parts[-3] in classes: # In case of extra subdir like 'PNG'
        return parts[-3]
    
    # Fallback: check if any class name is in the path
    for cls in classes:
        if cls in parts:
            return cls
            
    raise ValueError(f"Could not extract label from path: {image_path}")

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
                        {"type": "image", "image": image_path}, # Pass path to qwen_vl_utils/processor
                        {"type": "text", "text": "Describe this image."}, # Dummy text, we only need vision features
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
        # Qwen3-VL
        text = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        
        # Qwen3 might handle images differently in processor call compared to Qwen2.5
        # Based on extract_vision_embedding.py:
        inputs = processor(
            text=text,
            images=images, # Pass PIL images
            padding=True,
            return_tensors="pt",
        )
    else:
        # Qwen2.5-VL
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
        # Forward pass
        # Both Qwen2.5 and Qwen3 use the same signature for visual model in this context
        image_embeds = visual_model(pixel_values, grid_thw=image_grid_thw)
        
        # Handle tuple return if any (Qwen3 might return tuple)
        if isinstance(image_embeds, tuple):
            image_embeds = image_embeds[0]

    # Split features back to per-image
    # Calculate tokens per image
    merge_size = visual_model.spatial_merge_size
    
    # Calculate split sizes
    # image_grid_thw is (batch, 3) -> (t, h, w)
    # tokens = (t * h * w) / (merge_size^2)
    split_sizes = (torch.prod(image_grid_thw, dim=1) // (merge_size**2)).tolist()
    
    # Check for CLS token
    total_tokens = image_embeds.shape[0]
    expected_tokens = sum(split_sizes)
    
    assert total_tokens == expected_tokens, f"Total tokens {total_tokens} does not match expected tokens {expected_tokens}"
    if total_tokens == expected_tokens:
        # No CLS token
        features = list(torch.split(image_embeds, split_sizes))
    elif total_tokens == expected_tokens + len(split_sizes):
        # 1 CLS token per image
        # We need to be careful about where the CLS token is.
        # Usually it's appended or prepended.
        # Let's assume standard Qwen VL behavior:
        # If there is a CLS token, we might want to include it or exclude it.
        # For "vision only" features, usually we want the patch embeddings.
        # But if we just split, we need to know the order.
        
        # Let's try to split assuming they are contiguous blocks per image.
        # If 1 CLS token is added per image, the split size should be +1?
        # Or is it global?
        
        # In extract_vision_embedding_qwen2_5_vl.py, it printed:
        # "Actual output tokens: {actual_tokens}"
        # If they match, no CLS.
        
        # If there IS a mismatch, we need to handle it.
        # For now, let's assume the split_sizes need to be adjusted if we detect a consistent +1 per image.
        if total_tokens == sum([s + 1 for s in split_sizes]):
             split_sizes = [s + 1 for s in split_sizes]
             features = list(torch.split(image_embeds, split_sizes))
        else:
             print(f"Warning: Token count mismatch. Total: {total_tokens}, Expected from grid: {expected_tokens}. Batch size: {len(split_sizes)}")
             # Fallback: try to split evenly if possible? No, variable size.
             # Just return None to avoid saving bad data
             return None
    else:
        print(f"Warning: Token count mismatch. Total: {total_tokens}, Expected from grid: {expected_tokens}")
        return None

    # Post-process features: Global Average Pooling to get a single vector per image?
    # Or return the full spatial features?
    # The user asked for "extracted vision features", usually implies the full map or a pooled vector.
    # Given "classify/CCRCC/vlm_vision_only", and downstream usage (likely training a classifier),
    # a fixed size vector is easier, but spatial features are richer.
    # However, saving variable length tensors in a list is fine.
    # BUT, usually for "features", we might want GAP (Global Average Pooling).
    # Let's check `pfm/train_ccrcc.py` (from context) to see what it expects.
    # It says "extract image features using these models and then train a simple linear classifier".
    # Linear classifier usually needs fixed dim input.
    # So GAP is likely required.
    
    pooled_features = []
    for feat in features:
        # feat shape: (num_tokens, hidden_dim)
        # Mean pooling
        pooled = feat.mean(dim=0) # (hidden_dim)
        pooled_features.append(pooled.cpu())
        
    return pooled_features

def process_dataset(model_key, split_name):
    json_path = os.path.join(DATA_ROOT, f"{split_name}.json")
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    print(f"Processing {split_name} data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create output dirs
    feat_dir = os.path.join(OUTPUT_ROOT, split_name, "feat")
    label_dir = os.path.join(OUTPUT_ROOT, split_name, "label")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    model, processor = load_model(model_key)
    model.eval()
    
    all_features = []
    all_labels = []
    
    batch_size = 8
    
    for i in tqdm(range(0, len(data), batch_size)):
        batch_items = data[i : i + batch_size]
        
        # Extract labels
        batch_labels = []
        for item in batch_items:
            # item['image_path']
            # item['gt_label'] might exist if we used the divide script correctly
            if 'gt_label' in item:
                batch_labels.append(item['gt_label'])
            else:
                # Fallback to path parsing
                try:
                    lbl = get_label_from_path(item['image_path'])
                    batch_labels.append(lbl)
                except Exception as e:
                    print(f"Error getting label for {item['image_path']}: {e}")
                    batch_labels.append("unknown")
        
        # Extract features
        features = extract_features(model, processor, batch_items, model_key)
        
        if features:
            all_features.extend(features)
            all_labels.extend(batch_labels)
        else:
            print(f"Skipping batch {i} due to extraction error")
            
    # Save results
    # Features as .pt
    feat_save_path = os.path.join(feat_dir, f"{model_key}.pt")
    torch.save(torch.stack(all_features), feat_save_path)
    print(f"Saved features to {feat_save_path}")
    
    # Labels as .json
    label_save_path = os.path.join(label_dir, f"{model_key}.json")
    with open(label_save_path, 'w') as f:
        json.dump(all_labels, f)
    print(f"Saved labels to {label_save_path}")
    
    # Cleanup
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

def main():
    for model_key in MODEL_PATHS.keys():
        print(f"\n\n{'='*20} Processing {model_key} {'='*20}")
        try:
            process_dataset(model_key, "train")
            process_dataset(model_key, "test")
        except Exception as e:
            print(f"Error processing {model_key}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
