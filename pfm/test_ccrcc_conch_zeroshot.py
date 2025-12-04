import os
import json
import torch
import sys
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from os.path import join as j_

# Ensure pfm is in path if running from pfm/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("..")

from conch.conch import create_model_from_pretrained
from conch.downstream.zeroshot_path import zero_shot_classifier, run_zeroshot

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
        # Return dict as expected by run_zeroshot/dataloding_post_process
        return {'img': image, 'label': label}

def main():
    parser = argparse.ArgumentParser(description="CCRCC Conch Zero-shot Classification")
    parser.add_argument("--dataroot", type=str, default="/data/dataset/classification/CCRCC/tissue_classification/", help="Path to CCRCC dataset")
    parser.add_argument("--results_dir", type=str, default="/data/ljd/Pathology_FM_LLM/expriment/classify/CCRCC/conch_zeroshot", help="Directory to save results")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    print("Loading CONCH model...")
    model_cfg = 'conch_ViT-B-16'
    checkpoint_path = '/data/ckpt/conch/pytorch_model.bin'
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        # You might want to handle this or let create_model_from_pretrained fail/download
    
    model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, device=device, force_image_size=224)
    model.eval()

    # Create dataset
    print(f"Loading dataset from {args.dataroot}...")
    # Note: Using the full dataset as test set for zero-shot as per typical zero-shot eval usage, 
    # or should we split? The user request implies testing, usually on a test set.
    # train_ccrcc.py does an 80/20 split. 
    # If we want to evaluate on the *test* split defined in train_ccrcc.py, we should replicate that split.
    # However, for zero-shot, often we just evaluate on the whole target set if it's a new dataset.
    # But let's follow train_ccrcc.py's logic to be consistent if this is intended as a test.
    # User said "test_ccrcc_conch_zeroshot.py", implying test.
    # Let's use the same split logic to ensure we are testing on the "test" set.
    
    from torch.utils.data import random_split
    full_dataset = CCRCCDataset(args.dataroot, transform=preprocess)
    if len(full_dataset) == 0:
        print(f"No images found in {args.dataroot}")
        return

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    # Fix seed for reproducibility if needed, but train_ccrcc doesn't seem to set one explicitly for split
    # We will just use the test_dataset part.
    _, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    print(f"Test dataset size: {len(test_dataset)}")
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Define class names and templates
    # User requested: ["blood", "cancer", "normal", "stroma"]
    # zero_shot_classifier expects list of lists
    classnames = [
        ["blood"],
        ["cancer"],
        ["normal"],
        ["stroma"]
    ]

    templates = [
        "CLASSNAME.",
        "a histopathological image showing CLASSNAME.",
        "an H&E stained image of CLASSNAME.",
        "an image of CLASSNAME.",
        "an example of CLASSNAME.",
        "CLASSNAME is shown.",
        "this is CLASSNAME."
    ]

    # Build zero-shot classifier
    print("Building zero-shot classifier...")
    zeroshot_weights = zero_shot_classifier(model, classnames, templates, device=device)
    print(f"Zero-shot weights shape: {zeroshot_weights.shape}")

    # Run zero-shot classification
    print("Running zero-shot classification...")
    results, dump = run_zeroshot(model, zeroshot_weights, test_dataloader, device, 
                                dump_results=True, metrics=['acc', 'bacc', 'weighted_f1', 'roc_auc'])

    # Output results
    print("\nResults:")
    print("=" * 40)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    results_file = os.path.join(args.results_dir, "conch_zeroshot_results.txt")
    
    with open(results_file, "w") as f:
        f.write("CCRCC Dataset Zero-Shot Classification Results\n")
        f.write("=" * 50 + "\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
            
    # Also save as json for easier parsing later if needed
    json_file = os.path.join(args.results_dir, "conch_zeroshot_results.json")
    with open(json_file, "w") as f:
        # Convert numpy/tensor values to float for json serialization
        json_results = {k: float(v) for k, v in results.items()}
        json.dump(json_results, f, indent=4)
        
    print(f"\nResults saved to {results_file} and {json_file}")

if __name__ == "__main__":
    main()
