import os
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..")
# loading all packages here to start
from uni.get_encoder import get_encoder, get_eval_transforms
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from uni.downstream.eval_patch_features.metrics import get_eval_metrics, print_metrics


class ChaoyangDataset(Dataset):
    """Chaoyang dataset."""
    
    def __init__(self, data_root, json_path, transform=None):
        """
        Args:
            data_root (string): Directory with all the images.
            json_path (string): Path to the json file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_root = data_root
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = os.path.join(self.data_root, self.data[idx]['name'])
        image = Image.open(img_path).convert('RGB')
        label = self.data[idx]['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset paths
    data_root = "/data/dataset/chaoyang/"
    train_json = os.path.join(data_root, "train.json")
    test_json = os.path.join(data_root, "test.json")

    # Downloading UNI weights + Creating Model
    print("Loading UNI model...")
    model, transform = get_encoder(), get_eval_transforms()
    model.eval()
    model.to(device)

    # Create datasets
    print("Loading datasets...")
    train_dataset = ChaoyangDataset(data_root, train_json, transform=transform)
    test_dataset = ChaoyangDataset(data_root, test_json, transform=transform)

    # Create data loaders
    batch_size = 256
    num_workers = 8
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Extract features
    print("Extracting train features...")
    train_features = extract_patch_features_from_dataloader(model, train_dataloader)
    
    print("Extracting test features...")
    test_features = extract_patch_features_from_dataloader(model, test_dataloader)

    # Convert to tensors
    train_feats = torch.Tensor(train_features['embeddings'])
    train_labels = torch.Tensor(train_features['labels']).type(torch.long)
    test_feats = torch.Tensor(test_features['embeddings'])
    test_labels = torch.Tensor(test_features['labels']).type(torch.long)

    print(f"Train features shape: {train_feats.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test features shape: {test_feats.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    # Train linear probe
    print("Training linear probe...")
    linprobe_eval_metrics, linprobe_dump = eval_linear_probe(
        train_feats=train_feats,
        train_labels=train_labels,
        valid_feats=None,
        valid_labels=None,
        test_feats=test_feats,
        test_labels=test_labels,
        max_iter=1000,
        verbose=True,
    )

    # Print results
    print("\nResults:")
    print_metrics(linprobe_eval_metrics)
    
    # Save results
    results_file = "/data/ljd/Pathology_FM_LLM/expriment/classify/chaoyang/uni_linear_probe_results.txt"
    with open(results_file, "w") as f:
        f.write("Chaoyang Dataset Linear Probe Results\n")
        f.write("=" * 40 + "\n")
        for k, v in linprobe_eval_metrics.items():
            if "report" not in k:
                f.write(f"{k}: {v:.4f}\n")
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()