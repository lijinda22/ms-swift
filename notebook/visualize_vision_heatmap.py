import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os

# Set model path (adjust as needed)
MODEL_PATH = "/data/ckpt/Qwen3-VL-2B-Instruct"

def visualize_features():
    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Load a sample image
    image_path = "/data/ljd/other/critic_img_seven.png"
    image = Image.open(image_path).convert('RGB')

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

    # Extract Visual Embeddings
    visual_model = model.model.visual
    pixel_values = inputs.pixel_values.type(visual_model.dtype)
    image_grid_thw = inputs.image_grid_thw
    
    print("Extracting features...")
    with torch.no_grad():
        # shape: (num_tokens, hidden_size)
        image_embeds, _ = visual_model(pixel_values, grid_thw=image_grid_thw)

    # Reshape features back to 2D grid
    # image_grid_thw is [1, T, H, W] or just [T, H, W] depending on batch?
    # Processor returns image_grid_thw as tensor of shape (batch_images, 3) usually.
    # Let's assume single image for visualization.
    
    grid_t, grid_h, grid_w = image_grid_thw[0].tolist()
    merge_size = visual_model.spatial_merge_size
    
    # Calculate output spatial dimensions
    h_feat = grid_h // merge_size
    w_feat = grid_w // merge_size
    
    print(f"Original Grid (Patches): {grid_h}x{grid_w}")
    print(f"Feature Map Grid (after merger): {h_feat}x{w_feat}")
    print(f"Total Tokens: {image_embeds.shape[0]}")
    
    if image_embeds.shape[0] != h_feat * w_feat:
        print(f"Warning: Token count mismatch! Expected {h_feat*w_feat}, got {image_embeds.shape[0]}")
        # Try to infer shape?
        # For now, assert strict match
        raise ValueError("Token count does not match grid calculation.")

    # Reshape: (H*W, D) -> (H, W, D)
    # Validating layout: usually row-major.
    try:
        feature_map = image_embeds.reshape(h_feat, w_feat, -1).float().cpu().numpy()
    except Exception as e:
        print(f"Reshape failed: {e}")
        return

    # --- Visualization 1: Activation Heatmap (L2 Norm) ---
    feature_norm = np.linalg.norm(feature_map, axis=2)
    feature_norm = (feature_norm - feature_norm.min()) / (feature_norm.max() - feature_norm.min())

    # Resize heatmap to original image size for overlay
    # We use PIL for high-quality resizing
    heatmap_img = Image.fromarray((feature_norm * 255).astype(np.uint8))
    heatmap_resized = heatmap_img.resize(image.size, resample=Image.BICUBIC)
    heatmap_resized_np = np.array(heatmap_resized) / 255.0

    # Apply colormap
    cmap = plt.get_cmap('viridis')
    heatmap_colored = cmap(heatmap_resized_np)[:, :, :3] # Drop alpha if exists

    # Overlay: 0.5 * Image + 0.5 * Heatmap
    image_np = np.array(image).astype(float) / 255.0
    overlay = 0.6 * image_np + 0.4 * heatmap_colored

    plt.figure(figsize=(20, 6))
    
    plt.subplot(1, 4, 1)
    plt.title(f"Original Image\n{image.size}")
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title(f"Feature L2 Norm\nGrid: {h_feat}x{w_feat}")
    plt.imshow(feature_norm, cmap='viridis')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Overlay")
    plt.imshow(np.clip(overlay, 0, 1))
    plt.axis('off')

    # --- Visualization 2: K-Means Clustering (Semantic Segmentation) ---
    # Alternative to PCA: Group features into K clusters to show semantic regions
    print("Computing K-Means...")
    from sklearn.cluster import KMeans
    h, w, d = feature_map.shape
    features_flat = feature_map.reshape(-1, d)
    
    # Number of clusters (semantic regions)
    num_clusters = 6
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_flat)
    
    # Map labels to colors using a qualitative colormap
    cmap_clusters = plt.get_cmap('tab10', num_clusters)
    cluster_img_colored = cmap_clusters(cluster_labels)[:, :3] # (N, 3) RGB
    cluster_img = cluster_img_colored.reshape(h, w, 3)
    
    # Resize to original size
    cluster_pil = Image.fromarray((cluster_img * 255).astype(np.uint8))
    cluster_resized = cluster_pil.resize(image.size, resample=Image.NEAREST) # Nearest to keep distinct regions

    plt.subplot(1, 4, 4)
    plt.title(f"Semantic K-Means (K={num_clusters})")
    plt.imshow(cluster_resized)
    plt.axis('off')
    
    output_path = "/data/ljd/other/vision_heatmap.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_features()
