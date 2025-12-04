import argparse
import os
import sys
import time
import json
import torch
import logging
from os.path import join as j_
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Ensure pfm is in path if running from pfm/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("..")
# Imports from uni package
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from uni.downstream.eval_patch_features.metrics import print_metrics
from uni.get_encoder.get_encoder import get_encoder_uni, get_eval_transforms_uni, get_encoder_uni2, get_eval_transforms_uni2
from uni.get_encoder.conchv1_5 import create_model_from_pretrained as create_conchv1_5
from conch.conch import create_model_from_pretrained as create_conch


logging.basicConfig(level=logging.INFO)

# --- Dataset ---

class CCRCCDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.classes = ["blood", "cancer", "normal", "stroma"]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = j_(root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(
                        (j_(cls_dir, fname), self.class_to_idx[cls_name])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def get_ccrcc_dataloaders(transform, dataroot):
    """Dataloader for CCRCC, with random 80/20 split."""
    print("Setting up CCRCC dataloaders with random split...")

    full_dataset = CCRCCDataset(dataroot, transform)
    if not full_dataset.samples:
        print(f"No images found for specified classes in '{dataroot}'")
        return None, None

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader

# --- Evaluation ---

def run_evaluation(
    dataset_name, train_dataloader, test_dataloader, model, device, results_dir
):
    """Runs the full feature extraction and evaluation pipeline."""
    print(f"\n===== Running evaluation for: {dataset_name} =====")

    if train_dataloader is None or test_dataloader is None:
        print(f"Skipping {dataset_name} due to data loading issues.")
        return

    print("Starting feature extraction...")
    start_time = time.time()

    with torch.no_grad():
        train_features = extract_patch_features_from_dataloader(model, train_dataloader)
        test_features = extract_patch_features_from_dataloader(model, test_dataloader)

    train_feats = torch.Tensor(train_features["embeddings"])
    train_labels = torch.Tensor(train_features["labels"]).type(torch.long)
    test_feats = torch.Tensor(test_features["embeddings"])
    test_labels = torch.Tensor(test_features["labels"]).type(torch.long)

    elapsed = time.time() - start_time
    print(f"Feature extraction finished in {elapsed:.03f} seconds.")

    print("\nStarting linear probe evaluation...")

    linprobe_eval_metrics, _ = eval_linear_probe(
        train_feats=train_feats,
        train_labels=train_labels,
        valid_feats=None,
        valid_labels=None,
        test_feats=test_feats,
        test_labels=test_labels,
        max_iter=1000,
        verbose=False,
    )

    print(f"\n--- Results for {dataset_name} ---")
    print_metrics(linprobe_eval_metrics)
    print("---------------------------------------\n")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results_path = j_(results_dir, "evaluation_metrics.json")
    print(f"Saving results to {results_path}")
    with open(results_path, "w") as f:
        json.dump(linprobe_eval_metrics, f, indent=4)

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Train/Evaluate on CCRCC with various models")
    parser.add_argument("--models", type=str, nargs="+", default=["conch"], help="Models to use (default: all)")
    # parser.add_argument("--models", type=str, nargs="+", default=["uni", "uni2", "conch", "conchv1.5"], help="Models to use (default: all)")
    parser.add_argument("--dataroot", type=str, default="/data/dataset/classification/CCRCC/tissue_classification/", help="Path to CCRCC dataset")
    parser.add_argument("--results_dir", type=str, default="/data/ljd/Pathology_FM_LLM/expriment/classify", help="Base results directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    models_to_run = args.models
    print(f"Models to run: {models_to_run}")

    for model_name in models_to_run:
        print(f"\n{'='*30}")
        print(f"Processing model: {model_name}")
        print(f"{'='*30}\n")
        
        model = None
        transform = None

        if model_name == "uni":
            print("Loading UNI model...")
            transform = get_eval_transforms_uni()
            model = get_encoder_uni()
        
        elif model_name == "uni2":
            print("Loading UNI2 model...")
            transform = get_eval_transforms_uni2()
            model = get_encoder_uni2()
        
        elif model_name == "conchv1.5":
            print("Loading Conch v1.5 model...")
            # Uses default path in conchv1_5.py
            model, transform = create_conchv1_5()
        
        elif model_name == "conch":
            print("Loading Conch model...")
            if create_conch is None:
                raise ImportError("Conch module not found. Please ensure pfm/conch is accessible.")
            
            checkpoint_path = '/data/ckpt/conch/pytorch_model.bin'
            model_cfg = 'conch_ViT-B-16'
            model, transform = create_conch(model_cfg=model_cfg, checkpoint_path=checkpoint_path)

        if model is None:
            print(f"Skipping {model_name}: Model loading failed or unknown model name.")
            continue

        model.to(device)
        model.eval()
        print("Model and transforms loaded.")

        results_dir = j_(args.results_dir, "CCRCC", model_name)
        
        train_dl, test_dl = get_ccrcc_dataloaders(transform=transform, dataroot=args.dataroot)
        
        run_evaluation(
            dataset_name="CCRCC",
            train_dataloader=train_dl,
            test_dataloader=test_dl,
            model=model,
            device=device,
            results_dir=results_dir,
        )
        
        # Cleanup
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()


"""
可以使用Qwen2.5-VL-7b, Patho-R1, Lingshu, 这3个qwen2.5-vl模型, 以及LLaVA-Med, Qwen3-VL来对测试集评估

3个都是 Qwen2.5-VL 模型, 经过不同微调训练得到的, 原生模型和微调的领域模型
/data/ckpt/Qwen2.5-VL-7B-Instruct/
/data/ckpt/Patho-R1-7B/
/data/ckpt/Lingshu-7B/
/data/ckpt/Lingshu-32B/

Qwen3-VL: /data/ckpt/Qwen3-VL-2B-Instruct/

MLLM: LLaVA-Med 


"""