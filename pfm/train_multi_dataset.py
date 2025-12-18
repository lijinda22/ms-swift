import torch
import os
import time
import json
import argparse
from os.path import join as j_
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import logging
import sys

# Ensure module access
sys.path.append("..")
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from uni.downstream.eval_patch_features.metrics import print_metrics
from uni.get_encoder.conchv1_5 import create_model_from_pretrained as create_conchv1_5
from uni.get_encoder.get_encoder import get_encoder_uni, get_eval_transforms_uni, get_encoder_uni2, get_encoder_virchow2, get_eval_transforms_virchow2
from conch.conch import create_model_from_pretrained as create_conch

logging.basicConfig(level=logging.INFO)

# --- Config ---
DATASETS_BASE_DIR = "/data/ljd/Pathology_FM_LLM/expriment/classify"
RESULTS_BASE_DIR = "/data/ljd/Pathology_FM_LLM/expriment/classify/results_linear_probe"

DATASETS_CONFIG = [
    {"name": "CCRCC", "path": j_(DATASETS_BASE_DIR, "CCRCC")},
    {"name": "BreaKHis", "path": j_(DATASETS_BASE_DIR, "BreaKHis")},
    {"name": "chaoyang", "path": j_(DATASETS_BASE_DIR, "chaoyang")},
    {"name": "crc100k", "path": j_(DATASETS_BASE_DIR, "crc100k")},
    {"name": "CRC_MSI", "path": j_(DATASETS_BASE_DIR, "CRC_MSI")},
    {"name": "PanCancer-TIL", "path": j_(DATASETS_BASE_DIR, "PanCancer-TIL")},
]

# --- Unified Dataset ---
class UnifiedJsonDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.transform = transform
        json_path = j_(root, f"{split}.json")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        if not self.data:
            raise ValueError(f"Dataset at {json_path} is empty.")

        # Extract classes from metadata if available, otherwise infer from data
        # Assuming the generated JSONs have "gt_label"
        # We need a consistent class ordering. 
        # For simplicity, we sort unique labels found in the dataset.
        # Ideally, train and test should share the same class mapping. 
        # We'll build the mapping dynamically from the data, but this risks mismatch if test has missing classes.
        # However, our divide script ensures coverage or fixed lists. 
        # To be safe, we might need a way to pass classes. 
        # But wait, the divide script put "classes" list in EACH item. 
        # We can just read the classes from the first item!
        
        self.classes = sorted(self.data[0]["classes"])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image_path"]
        label_str = item["gt_label"]
        label = self.class_to_idx[label_str]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy
            image = Image.new("RGB", (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(dataset_name, dataset_path, transform):
    print(f"Loading {dataset_name} from {dataset_path}...")
    try:
        train_ds = UnifiedJsonDataset(dataset_path, split="train", transform=transform)
        test_ds = UnifiedJsonDataset(dataset_path, split="test", transform=transform)
        
        # Verify classes match
        if train_ds.classes != test_ds.classes:
            print(f"Warning: Class mismatch for {dataset_name}. Train: {train_ds.classes}, Test: {test_ds.classes}")
            # Proceed assume intersection or sorted match? 
            # If our divide script is correct, they must match.
        
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        
        return train_loader, test_loader
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        return None, None

def run_evaluation(dataset_name, train_dl, test_dl, model, device, results_dir):
    print(f"\n===== Running evaluation for: {dataset_name} =====")
    
    if not train_dl or not test_dl:
        return

    os.makedirs(results_dir, exist_ok=True)

    # if saved model exists, return
    if os.path.exists(j_(results_dir, "metrics.json")):
        print(f"Metrics already exists for {dataset_name}. Skipping evaluation.")
        return
    
    # Feature Extraction
    print("Extracting features...")
    with torch.no_grad():
        train_res = extract_patch_features_from_dataloader(model, train_dl)
        test_res = extract_patch_features_from_dataloader(model, test_dl)
        
    train_feats = torch.Tensor(train_res["embeddings"])
    train_labels = torch.Tensor(train_res["labels"]).long()
    test_feats = torch.Tensor(test_res["embeddings"])
    test_labels = torch.Tensor(test_res["labels"]).long()
    
    print(f"Features extracted. Train: {train_feats.shape}, Test: {test_feats.shape}")
    
    # Linear Probe
    print("Training linear probe...")
    metrics, dump = eval_linear_probe(
        train_feats=train_feats,
        train_labels=train_labels,
        valid_feats=None, 
        valid_labels=None,
        test_feats=test_feats,
        test_labels=test_labels,
        max_iter=1000,
        verbose=True
    )
    
    # Save Metrics
    print_metrics(metrics)
    with open(j_(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save Model
    if "logreg" in dump:
        torch.save(dump["logreg"], j_(results_dir, "linear_probe_model.pth"))
        print(f"Model saved to {j_(results_dir, 'linear_probe_model.pth')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["virchow2"], help="Models to run")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    for model_name in args.models:
        print(f"\n\n{'#'*40}\nProcessing Model: {model_name}\n{'#'*40}")
        
        # Load Model
        model = None
        transform = None
        
        try:
            if model_name == "uni":
                model = get_encoder_uni()
                transform = get_eval_transforms_uni()
            elif model_name == "uni2":
                model = get_encoder_uni2()
                transform = get_eval_transforms_uni()
            elif model_name == "conchv1.5":
                model, transform = create_conchv1_5()
            elif model_name == "conch":
                 checkpoint_path = '/data/ckpt/conch/pytorch_model.bin'
                 model, transform = create_conch(model_cfg='conch_ViT-B-16', checkpoint_path=checkpoint_path)
            elif model_name == "virchow2":
                 model = get_encoder_virchow2()
                 transform = get_eval_transforms_virchow2()
            
            if model:
                model.to(device)
                model.eval()
            else:
                print(f"Failed to load {model_name}")
                continue
                
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue

        # Run Datasets
        for ds_conf in DATASETS_CONFIG:
            train_dl, test_dl = get_dataloaders(ds_conf["name"], ds_conf["path"], transform)
            if train_dl and test_dl:
                res_dir = j_(RESULTS_BASE_DIR, ds_conf["name"], model_name)
                run_evaluation(ds_conf["name"], train_dl, test_dl, model, device, res_dir)
        
        # Cleanup
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
