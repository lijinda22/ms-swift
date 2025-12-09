import os
import json
import torch
import sys
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from os.path import join as j_

# Ensure pfm is in path
sys.path.append("..")

from conch.conch import create_model_from_pretrained
from conch.downstream.zeroshot_path import zero_shot_classifier, run_zeroshot

# --- Config ---
DATASETS_BASE_DIR = "/data/ljd/Pathology_FM_LLM/expriment/classify"
RESULTS_BASE_DIR = "/data/ljd/Pathology_FM_LLM/expriment/classify/results_zeroshot"

DATASETS_CONFIG = [
    {"name": "CCRCC", "path": j_(DATASETS_BASE_DIR, "CCRCC")},
    {"name": "BreaKHis", "path": j_(DATASETS_BASE_DIR, "BreaKHis")},
    {"name": "chaoyang", "path": j_(DATASETS_BASE_DIR, "chaoyang")},
    {"name": "crc100k", "path": j_(DATASETS_BASE_DIR, "crc100k")},
    {"name": "CRC_MSI", "path": j_(DATASETS_BASE_DIR, "CRC_MSI")},
    {"name": "PanCancer-TIL", "path": j_(DATASETS_BASE_DIR, "PanCancer-TIL")},
]

TEMPLATES = [
    "CLASSNAME.",
    "a histopathological image showing CLASSNAME.",
    "an H&E stained image of CLASSNAME.",
    "an image of CLASSNAME.",
    "an example of CLASSNAME.",
    "CLASSNAME is shown.",
    "this is CLASSNAME."
]

class UnifiedJsonDataset(Dataset):
    def __init__(self, root, split="test", transform=None):
        self.transform = transform
        json_path = j_(root, f"{split}.json")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        if not self.data:
            raise ValueError(f"Dataset at {json_path} is empty.")
            
        self.classes = sorted(self.data[0]["classes"])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image_path"]
        label = self.class_to_idx[item["gt_label"]]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224))
            
        if self.transform:
            image = self.transform(image)
        # Expected dict for zeroshot pipeline
        return {'img': image, 'label': label}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load Model
    print("Loading Conch model...")
    checkpoint_path = '/data/ckpt/conch/pytorch_model.bin'
    model, transform = create_model_from_pretrained(model_cfg='conch_ViT-B-16', checkpoint_path=checkpoint_path, device=device)
    model.eval()
    
    for ds_conf in DATASETS_CONFIG:
        name = ds_conf["name"]
        print(f"\nProcessing {name}...")
        
        try:
            ds = UnifiedJsonDataset(ds_conf["path"], split="test", transform=transform)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue
            
        # Classnames list of lists
        classnames = [[c] for c in ds.classes]
        print(f"Classes: {ds.classes}")
        
        # Build Classifier
        print("Building classifier...")
        zs_weights = zero_shot_classifier(model, classnames, TEMPLATES, device=device)
        
        # Run
        print("Running evaluation...")
        results, _ = run_zeroshot(model, zs_weights, loader, device, metrics=['acc', 'bacc', 'weighted_f1', 'roc_auc'])
        
        print(f"Results: {results}")
        
        # Save
        res_dir = j_(RESULTS_BASE_DIR, name)
        os.makedirs(res_dir, exist_ok=True)
        with open(j_(res_dir, "metrics.json"), "w") as f:
            # Convert to float
            clean_res = {k: float(v) for k, v in results.items()}
            json.dump(clean_res, f, indent=4)
            
    print("\nAll Done.")

if __name__ == "__main__":
    main()
