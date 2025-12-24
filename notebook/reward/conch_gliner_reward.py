import os
import sys
import json
import torch
from PIL import Image
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from gliner import GLiNER

# Add project root to sys.path
sys.path.append("../../")

from pfm.conch.conch.factory import create_model_from_pretrained
from pfm.conch.conch.custom_tokenizer import get_tokenizer, tokenize

def load_gliner_model(model_name="/data/ckpt/camembert-bio-gliner-v0.1/"):
    print(f"Loading GLiNER model: {model_name}...")
    model = GLiNER.from_pretrained(model_name)
    return model

def extract_morphology(text, model, labels=None):
    if labels is None:
        labels = ["morphological feature", "cell structure", "tissue architecture", "abnormality", "pathology description"]
    
    # Truncate text to avoid model length limit errors (CamemBERT usually 512 tokens)
    # 2000 chars is roughly 400-600 tokens depending on language
    if len(text) > 2000:
        text = text[:2000]

    try:
        # Use flat_ner=True to encourage merging adjacent entities
        entities = model.predict_entities(text, labels, flat_ner=True)
    except Exception as e:
        print(f"Warning: GLiNER prediction failed for text '{text[:30]}...': {e}")
        return []
    
    # 1. Deduplication (case-insensitive)
    unique_entities = []
    seen = set()
    
    # Sort by length descending to process longer entities first (optional but helpful for some logic)
    # Here we just collect first.
    raw_texts = []
    for entity in entities:
        text_val = entity['text'].strip()
        if not text_val:
            continue
        if text_val.lower() not in seen:
            seen.add(text_val.lower())
            raw_texts.append(text_val)

    # 2. Substring filtering: 
    # If "irregular nuclear contours" is extracted, we should probably discard "nuclear contours" if it offers no new info.
    # We keep an entity if it is NOT a substring of any OTHER entity.
    final_keywords = []
    for i, t1 in enumerate(raw_texts):
        is_substring = False
        for j, t2 in enumerate(raw_texts):
            if i != j and t1.lower() in t2.lower():
                is_substring = True
                break
        
        if not is_substring:
            final_keywords.append(t1)

    return final_keywords

def load_conch_model(checkpoint_path='/data/ckpt/conch/pytorch_model.bin', device='cuda'):
    print(f"Loading Conch model from {checkpoint_path}...")
    model_cfg = 'conch_ViT-B-16'
    model, transform = create_model_from_pretrained(
        model_cfg=model_cfg, 
        checkpoint_path=checkpoint_path, 
        device=device
    )
    model.eval()
    return model, transform

def main():
    # Configuration
    data_path = "/data/ljd/VLM-R1/dataset/sft/pathgen_instruct_close_cot_9144.jsonl"
    conch_ckpt = "/data/ckpt/conch/pytorch_model.bin"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = "/data/ljd/Pathology_FM_LLM/expriment/reward/conch_gliner_results.jsonl"

    # Load models
    gliner_model = load_gliner_model()
    if gliner_model is None:
        return

    conch_model, conch_transform = load_conch_model(conch_ckpt, device)

    print(f"Processing data from {data_path}...")
    
    results = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Process randomly sampled 100 lines
    sample_size = 1000
    if len(lines) > sample_size:
        print(f"Randomly sampling {sample_size} items from {len(lines)} total items...")
        lines = random.sample(lines, sample_size)
    else:
        print(f"Dataset size ({len(lines)}) is smaller than requested sample size ({sample_size}). Using all items.")

    processed_count = 0
    similarity_scores = []
    
    with torch.no_grad():
        for line in tqdm(lines):
            item = json.loads(line)
            image_path = item.get("image_path")
            reasoning_cot = item.get("reasoning_cot_lingshu32b")
            
            if not image_path or not reasoning_cot:
                continue
            
            # Check if image exists
            if not os.path.exists(image_path):
                # Try to fix path if it's relative or has a prefix issue
                # For this environment, assuming absolute paths are correct or need minimal adjustment
                pass 

            # 1. Extract morphology
            morph_features = extract_morphology(reasoning_cot, gliner_model)
            morph_text = ", ".join(morph_features)
            
            if not morph_text:
                morph_text = reasoning_cot # Fallback to full text if no features found? 

            # 2. Encode Image
            # No try-except as requested
            image = Image.open(image_path).convert('RGB')
            image_tensor = conch_transform(image).unsqueeze(0).to(device)
            image_emb = conch_model.encode_image(image_tensor, proj_contrast=True, normalize=True)

            # 3. Encode Text (Extracted Features)
            # Tokenize the text using Conch's custom tokenizer
            tokenizer = get_tokenizer()
            text_tokens = tokenize(texts=[morph_text], tokenizer=tokenizer).to(device)
            
            # Encode text
            # encode_text returns normalized embeddings by default (if normalize=True)
            text_emb = conch_model.encode_text(text_tokens)
            
            # 4. Calculate Cosine Similarity
            # Both embeddings are already normalized (L2 norm = 1)
            # So cosine similarity is just the dot product
            # image_emb: [1, D], text_emb: [1, D]
            
            # similarity = (image_emb @ text_emb.T).item() 改为计算余弦相识度
            raw_similarity = torch.nn.functional.cosine_similarity(image_emb, text_emb, dim=1).item()
            
            # Assert range [-1, 1] with slight tolerance for float precision
            assert -1.1 <= raw_similarity <= 1.1, f"Cosine similarity {raw_similarity} out of expected range [-1, 1]"
            
            # Scale to [0, 1]
            similarity = (raw_similarity + 1) / 2
            
            similarity_scores.append(similarity)
            
            result = {
                "image_path": image_path,
                "slide_id": item.get("slide_id", ""),
                "request_id": item.get("request_id", ""),
                "extracted_morphology": morph_features,
                "extracted_text_used": morph_text,
                "cosine_similarity": similarity, 
            }
            results.append(result)
            print(f"[{processed_count+1}] Similarity: {similarity:.4f} | Features: {morph_text[:50]}...")
            processed_count += 1

    # Save results
    print(f"Saving {len(results)} results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    # Plot distribution
    plot_dir = "/data/ljd/Pathology_FM_LLM/expriment/reward/"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "similarity_distribution.png")
    
    print(f"Plotting distribution to {plot_path}...")
    plt.figure(figsize=(10, 6))
    plt.hist(similarity_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Image-Text Cosine Similarity (GLiNER + Conch)')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.5)
    plt.text(0.05, 0.95, f'Mean: {np.mean(similarity_scores):.4f}\nStd: {np.std(similarity_scores):.4f}\nN={len(similarity_scores)}', 
             transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
            
    print("Done.")

if __name__ == "__main__":
    main()

