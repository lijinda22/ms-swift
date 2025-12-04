import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
sys.path.append("..")
from conch.conch import create_model_from_pretrained
from conch.downstream.zeroshot_path import zero_shot_classifier, run_zeroshot
from tqdm import tqdm
import numpy as np


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
    test_json = os.path.join(data_root, "test.json")

    # Create model
    print("Loading CONCH model...")
    model_cfg = 'conch_ViT-B-16'
    # 根据实际模型路径修改此处
    checkpoint_path = '/data/ckpt/conch/pytorch_model.bin'  
    model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, device=device, force_image_size=224)
    _ = model.eval()

    # Create dataset
    print("Loading test dataset...")
    test_dataset = ChaoyangDataset(data_root, test_json, transform=preprocess)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 定义类别名称和提示模板
    # 类别: 0: Normal mucosa; 1: Serrated lesions; 2: Adenocarcinoma; 3: Adenoma
    classnames = [
        ["normal mucosa", "healthy mucosa", "non-cancerous mucosa"],
        ["serrated lesions", "serrated polyp", "serrated lesion"],
        ["adenocarcinoma", "colorectal adenocarcinoma", "malignant glandular tumor"],
        ["adenoma", "adenomatous polyp", "benign glandular tumor"]
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

    # 构建zero-shot分类器
    print("Building zero-shot classifier...")
    zeroshot_weights = zero_shot_classifier(model, classnames, templates, device=device)
    print(f"Zero-shot weights shape: {zeroshot_weights.shape}")

    # 运行zero-shot分类
    print("Running zero-shot classification...")
    results, dump = run_zeroshot(model, zeroshot_weights, test_dataloader, device, 
                                dump_results=True, metrics=['acc', 'bacc', 'weighted_f1', 'roc_auc'])

    # 输出结果
    print("\nResults:")
    print("=" * 40)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # 保存结果到文件
    results_file = "/data/ljd/Pathology_FM_LLM/expriment/classify/chaoyang/conch_zeroshot_results.txt"
    with open(results_file, "w") as f:
        f.write("Chaoyang Dataset Zero-Shot Classification Results\n")
        f.write("=" * 50 + "\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()