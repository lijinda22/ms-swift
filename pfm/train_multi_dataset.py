import torch
import torchvision
import os
import time
import json
from os.path import join as j_
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import logging
import timm
from torchvision import transforms

import sys

sys.path.append("..")
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from uni.downstream.eval_patch_features.metrics import print_metrics
from uni.get_encoder.conchv1_5 import create_model_from_pretrained
from uni.get_encoder.get_encoder import get_encoder_uni, get_eval_transforms_uni

# --- New model/transform loaders from user ---
logging.basicConfig(level=logging.INFO)

# --- Custom Dataset for Chaoyang ---
class JsonDataset(Dataset):
    """Custom Dataset for datasets with labels in a JSON file."""

    def __init__(self, root, json_path, transform=None):
        self.root = root
        self.transform = transform
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.classes = sorted(list(set([item["label"] for item in self.data])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = j_(self.root, item["name"])
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: File not found {img_path}. Skipping.")
            return torch.randn(3, 224, 224), -1
        label = self.class_to_idx.get(item["label"], -1)

        if self.transform:
            image = self.transform(image)

        return image, label


# --- Data Loading Functions ---


def get_chaoyang_dataloaders(transform, dataroot):
    """Dataloader for the Chaoyang dataset."""
    print("Setting up Chaoyang dataloaders...")
    train_json_path = j_(dataroot, "train.json")
    test_json_path = j_(dataroot, "test.json")

    if not (os.path.exists(train_json_path) and os.path.exists(test_json_path)):
        print(f"JSON files not found in '{dataroot}'")
        return None, None

    train_dataset = JsonDataset(
        root=dataroot, json_path=train_json_path, transform=transform
    )
    test_dataset = JsonDataset(
        root=dataroot, json_path=test_json_path, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader


def get_ccrcc_dataloaders(transform, dataroot):
    """Dataloader for CCRCC, with random 80/20 split."""
    print("Setting up CCRCC dataloaders with random split...")

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


def get_image_folder_dataloaders(transform, dataroot, train_dir_name, test_dir_name):
    """Generic dataloader for ImageFolder-compatible datasets."""
    print(f"Setting up ImageFolder dataloaders for {dataroot}...")
    train_dir = j_(dataroot, train_dir_name)
    test_dir = j_(dataroot, test_dir_name)

    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        print(f"Train/Test directories not found in '{dataroot}'")
        return None, None

    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader


# --- Core Logic ---


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


def main():
    """Main function to orchestrate evaluation on all datasets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_name = "conchv1.5"
    # model_name in "conchv1.5" "uni" "uni2" "conch"

    if model_name == "uni": 
        transform = get_eval_transforms_uni()
        model = get_encoder_uni()
    elif model_name == "conchv1.5":
        model, transform = create_model_from_pretrained()
    model.to(device)
    model.eval()
    print("Model and transforms loaded.")

    base_results_dir = "/data/ljd/Pathology_FM_LLM/expriment/classify"

    datasets_config = [
        {
            "name": "CCRCC",
            "path": "/data/dataset/classification/CCRCC/tissue_classification/",
            "loader": get_ccrcc_dataloaders,
            "args": {},
        },
        {
            "name": "chaoyang",
            "path": "/data/dataset/classification/chaoyang/",
            "loader": get_chaoyang_dataloaders,
            "args": {},
        },
        {
            "name": "CRC-100k",
            "path": "/data/dataset/classification/crc100k/",
            "loader": get_image_folder_dataloaders,
            "args": {
                "train_dir_name": "NCT-CRC-HE-100K-NONORM",
                "test_dir_name": "CRC-VAL-HE-7K",
            },
        },
        {
            "name": "CRC_MSI",
            "path": "/data/dataset/classification/CRC_MSI/",
            "loader": get_image_folder_dataloaders,
            "args": {"train_dir_name": "TRAIN", "test_dir_name": "TEST"},
        },
    ]

    for config in datasets_config:
        print(f"\n{'='*20}\nProcessing dataset: {config['name']}\n{'='*20}")

        results_dir = j_(base_results_dir, config["name"], model_name)

        loader_func = config["loader"]
        loader_args = config.get("args", {})

        train_dl, test_dl = loader_func(
            transform=transform, dataroot=config["path"], **loader_args
        )

        run_evaluation(
            dataset_name=config["name"],
            train_dataloader=train_dl,
            test_dataloader=test_dl,
            model=model,
            device=device,
            results_dir=results_dir,
        )


if __name__ == "__main__":
    main()
