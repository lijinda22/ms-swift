import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import sys
import os

# Set model path
MODEL_PATH = "/data/ckpt/Qwen2.5-VL-7B-Instruct"

def run_extraction():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load a sample image
    image_url = "/data/ljd/VLM-R1/ckpt/vlm-r1-rec-dataset/COCO_train2014_000000379143.jpg"
    if os.path.exists(image_url):
        image = Image.open(image_url)
    else:
        print(f"Image not found at {image_url}, using dummy white image.")
        image = Image.new('RGB', (224, 224), color='white')

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe dimensions."},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # 1. Extract Visual Embeddings
    # Qwen2.5-VL structure: model.model.visual
    visual_model = model.model.visual
    
    # Get inputs for visual model
    pixel_values = inputs.pixel_values
    image_grid_thw = inputs.image_grid_thw
    
    print("\n--- Visual Feature Extraction (Qwen2.5-VL) ---")
    print(f"Input Pixel Values Shape: {pixel_values.shape}")
    print(f"Image Grid (T, H, W): {image_grid_thw}")
    
    # Forward pass through vision tower
    with torch.no_grad():
        pixel_values = pixel_values.type(visual_model.dtype)
        # Qwen2.5-VL returns only hidden_states, unlike Qwen3-VL which returns a tuple
        image_embeds = visual_model(pixel_values, grid_thw=image_grid_thw)
        
    print(f"Extracted Visual Embeddings Shape (after projection): {image_embeds.shape}")
    
    # 2. Check for CLS Token
    merge_size = visual_model.spatial_merge_size
    print(f"Spatial Merge Size: {merge_size}")

    # Calculate expected tokens based on grid_thw
    # Qwen2.5-VL Logic: split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    
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
    
    # Check dimensions
    vision_hidden_size = visual_model.config.hidden_size
    merger_input_dim = vision_hidden_size * (merge_size ** 2)
    
    # Inspect MLP structure in Qwen2.5-VL Merger
    # self.mlp = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.GELU(), nn.Linear(self.hidden_size, dim))
    
    if hasattr(merger, 'mlp'):
        # Accessing first and last linear layers of the Sequential block
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
