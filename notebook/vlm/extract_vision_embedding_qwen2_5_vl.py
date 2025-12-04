import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import sys
import os

# Set model path
MODEL_PATH = "/data/ckpt/Lingshu-7B"

def run_extraction():
    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    image_url = "/data/dataset/classification/CRC_MSI/TRAIN/MSIH/TCGA-5M-AAT6-01Z-00-DX1.8834C952-14E3-4491-8156-52FC917BB014_(12089,43730).jpg"
    image = Image.open(image_url)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe dimensions."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    visual_model = model.model.visual
    pixel_values = inputs.pixel_values
    image_grid_thw = inputs.image_grid_thw
    print("\n--- Visual Feature Extraction (Qwen2.5-VL) ---")
    print(f"Input Pixel Values Shape: {pixel_values.shape}")
    print(f"Image Grid (T, H, W): {image_grid_thw}")
    with torch.no_grad():
        pixel_values = pixel_values.type(visual_model.dtype)
        image_embeds = visual_model(pixel_values, grid_thw=image_grid_thw)
    print(f"Extracted Visual Embeddings Shape (after projection): {image_embeds.shape}")
    merge_size = visual_model.spatial_merge_size
    print(f"Spatial Merge Size: {merge_size}")
    calculated_tokens = torch.sum(torch.prod(image_grid_thw, dim=1)).item() // (merge_size**2)
    actual_tokens = image_embeds.shape[0]
    
    print(f"Calculated expected tokens (from grid / merge_size^2): {calculated_tokens}")
    print(f"Actual output tokens: {actual_tokens}")
    
    if actual_tokens == calculated_tokens:
        print(">>> Conclusion: The extracted features DO NOT include a specialized CLS token.")
        print("    (The output token count matches exactly the spatial grid reduction).")
    elif actual_tokens == calculated_tokens + image_grid_thw.shape[0]:
        print(">>> Conclusion: The extracted features INCLUDE a CLS token (one per image).")
    else:
        print(f">>> Conclusion: Token count mismatch (Diff: {actual_tokens - calculated_tokens}). Check architecture details.")

    # 3. Projection Layer Mapping
    # The 'merger' in Qwen2.5-VL is Qwen2_5_VLPatchMerger
    merger = visual_model.merger
    
    vision_hidden_size = visual_model.config.hidden_size
    merger_input_dim = vision_hidden_size * (merge_size ** 2)
    if hasattr(merger, 'mlp'):
        first_linear = merger.mlp[0]
        last_linear = merger.mlp[2]
        proj_layer_in = first_linear.in_features
        proj_layer_out = last_linear.out_features
        print(f"\n--- Projection Layer (Merger) ---")
        print(f"Vision Model Hidden Size: {vision_hidden_size}")
        print(f"Merger Input Dimension (spatial concatenation): {merger_input_dim} (Checked against MLP first layer: {proj_layer_in})")
        print(f"Merger Output Dimension (Projection): {proj_layer_out}")
    else:
        print("\n--- Projection Layer (Merger) ---")
        print("Could not identify 'mlp' attribute in merger. Structure might differ from expected Qwen2_5_VLPatchMerger.")

    # Check LLM Text Embedding size
    llm_embed_dim = model.model.language_model.embed_tokens.embedding_dim
    print(f"LLM Text Embedding Dimension: {llm_embed_dim}")
    
    if hasattr(merger, 'mlp') and proj_layer_out == llm_embed_dim:
        print(">>> Verification Successful: The vision features are projected to match the LLM's word embedding space.")
    else:
        print(">>> Verification Failed or skipped: Projection output size check.")

if __name__ == "__main__":
    run_extraction()
