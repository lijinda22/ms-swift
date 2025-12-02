import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import sys

# Set model path (using the one from the context example)
# In a real run, ensure this path exists or points to "Qwen/Qwen2.5-VL-7B-Instruct" etc.
MODEL_PATH = "/data/ckpt/Qwen3-VL-2B-Instruct"

def run_extraction():
    print(f"Loading model from {MODEL_PATH}...")
    # Using trust_remote_code=True because Qwen3-VL might be custom code as indicated by the project structure
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Load a sample image (using a reliable URL or local file if available)
    # Using a placeholder dummy image if URL fetch fails
    image_url = "/data/ljd/VLM-R1/ckpt/vlm-r1-rec-dataset/COCO_train2014_000000379143.jpg"
    image = Image.open(image_url)
    # image = Image.new('RGB', (224, 224), color='white')

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
    # Qwen3VL structure: model.model.visual is the Qwen3VLVisionModel
    visual_model = model.model.visual
    
    # Get inputs for visual model
    pixel_values = inputs.pixel_values
    image_grid_thw = inputs.image_grid_thw
    
    print("\n--- Visual Feature Extraction ---")
    print(f"Input Pixel Values Shape: {pixel_values.shape}")
    print(f"Image Grid (T, H, W): {image_grid_thw}")
    
    # Forward pass through vision tower
    # The visual model's forward pass includes the transformer blocks AND the merger (projection)
    with torch.no_grad():
        # pixel_values needs to be cast to model dtype
        pixel_values = pixel_values.type(visual_model.dtype)
        # Note: The calling convention in Qwen3VLModel.get_image_features passes grid_thw
        image_embeds, deepstack_image_embeds = visual_model(pixel_values, grid_thw=image_grid_thw)
        
    print(f"Extracted Visual Embeddings Shape (after projection): {image_embeds.shape}")
    
    # 2. Check for CLS Token
    # grid_thw comes from the processor and represents the grid size of the patch embeddings *before* merging?
    # Let's verify with the output size.
    # visual_model.spatial_merge_size dictates the reduction.
    
    merge_size = visual_model.spatial_merge_size
    print(f"Spatial Merge Size: {merge_size}")

    # Calculate expected tokens based on grid_thw
    # image_grid_thw is (num_images, 3) where 3 is (T, H, W) of the features
    # Note: In Qwen2-VL/Qwen3-VL, grid_thw usually represents the shape of the features coming out of the ViT *before* the merger?
    # Or does it represent the input grid?
    # Let's look at Qwen3VLModel code logic:
    # split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    # This implies image_grid_thw is the size BEFORE merging.
    
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
    # The 'merger' in Qwen3VLVisionModel acts as the projection layer.
    merger = visual_model.merger
    
    # Check dimensions
    # The merger takes (hidden_size * merge_size^2) and projects to out_hidden_size
    # hidden_size of vision model
    vision_hidden_size = visual_model.config.hidden_size
    merger_input_dim = vision_hidden_size * (merge_size ** 2)
    
    # actual linear layer
    proj_layer_in = merger.linear_fc1.in_features
    proj_layer_out = merger.linear_fc2.out_features
    
    print(f"\n--- Projection Layer (Merger) ---")
    print(f"Vision Model Hidden Size: {vision_hidden_size}")
    print(f"Merger Input Dimension (spatial concatenation): {merger_input_dim} (Checked against layer: {proj_layer_in})")
    print(f"Merger Output Dimension (Projection): {proj_layer_out}")
    
    # Check LLM Text Embedding size
    llm_embed_dim = model.model.language_model.embed_tokens.embedding_dim
    print(f"LLM Text Embedding Dimension: {llm_embed_dim}")
    
    if proj_layer_out == llm_embed_dim:
        print(">>> Verification Successful: The vision features are projected to match the LLM's word embedding space.")
    else:
        print(">>> Verification Failed: Projection output size does not match LLM embedding size.")

if __name__ == "__main__":
    run_extraction()
