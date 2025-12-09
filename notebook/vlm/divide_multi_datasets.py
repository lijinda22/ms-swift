import os
import json
import csv
import glob
import random
from os.path import join as pjoin
from typing import List, Dict

# Config
OUTPUT_BASE = "/data/ljd/Pathology_FM_LLM/expriment/classify/"

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Saved {len(data)} items to {path}")

def format_question(classes: List[str], descriptions: Dict[str, str] = None):
    # Map classes to A, B, C, D...
    options_map = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, cls_name in enumerate(classes):
        options_map[letters[i]] = cls_name
    
    question_str = f'Which of the following classes does this pathology image belong to: {", ".join([f"{c}" for c in classes])}?\n'
    for letter, cls_name in options_map.items():
        desc = cls_name
        if descriptions and cls_name in descriptions:
            desc = descriptions[cls_name]
        question_str += f'{letter}. {desc}\n'
    
    return question_str.strip(), options_map

def process_ccrcc():
    print("Processing CCRCC...")
    dataroot = "/data/dataset/classification/CCRCC/tissue_classification/"
    output_dir = pjoin(OUTPUT_BASE, "CCRCC")
    
    classes = ["blood", "cancer", "normal", "stroma"]
    class_descriptions = {
        "blood": "Red blood cells",
        "cancer": "Renal cancer",
        "normal": "Normal renal",
        "stroma": "Stromal, including smooth muscle, fibrous stroma and blood vessels"
    }
    
    all_samples = []
    for cls_name in classes:
        cls_dir = pjoin(dataroot, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        images = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
            images.extend(glob.glob(pjoin(cls_dir, ext)))
            images.extend(glob.glob(pjoin(cls_dir, ext.upper())))
        
        for img_path in images:
            all_samples.append({"path": img_path, "label": cls_name})

    random.seed(42)
    random.shuffle(all_samples)
    train_size = int(0.8 * len(all_samples))
    train_data = all_samples[:train_size]
    test_data = all_samples[train_size:]
    
    question_text, options_map = format_question(classes, class_descriptions)
    reverse_map = {v: k for k, v in options_map.items()}

    def make_qa(samples):
        res = []
        for s in samples:
            res.append({
                "question": question_text,
                "answer": reverse_map[s["label"]],
                "image_path": s["path"],
                "gt_label": s["label"],
                "classes": classes
            })
        return res

    save_json(make_qa(train_data), pjoin(output_dir, "train.json"))
    save_json(make_qa(test_data), pjoin(output_dir, "test.json"))

def process_breakhis():
    print("Processing BreaKHis...")
    folds_csv = "/data/dataset/classification/BreaKHis/Folds.csv"
    data_root = "/data/dataset/classification/BreaKHis/BreaKHis_v1/"
    output_dir = pjoin(OUTPUT_BASE, "BreaKHis")
    
    classes = ["Benign tumor", "Malignant tumor"]
    question_text, options_map = format_question(classes)
    reverse_map = {v: k for k, v in options_map.items()}
    
    if not os.path.exists(folds_csv):
        print(f"Warning: {folds_csv} not found, skipping BreaKHis.")
        return

    train_samples, test_samples = [], []
    with open(folds_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            grp = row['grp'].strip()
            filename = row['filename'].strip()
            
            label = "Benign tumor" if 'benign' in filename.lower() else "Malignant tumor" if 'malignant' in filename.lower() else None
            if not label: continue
            
            item = {
                "question": question_text,
                "answer": reverse_map[label],
                "image_path": pjoin(data_root, filename),
                "gt_label": label,
                "classes": classes
            }
            if grp == 'train': train_samples.append(item)
            elif grp == 'test': test_samples.append(item)

    save_json(train_samples, pjoin(output_dir, "train.json"))
    save_json(test_samples, pjoin(output_dir, "test.json"))

def process_chaoyang():
    print("Processing Chaoyang...")
    input_base = "/data/dataset/classification/chaoyang/"
    output_dir = pjoin(OUTPUT_BASE, "chaoyang")
    
    label_map = {
        0: "Normal mucosa",
        1: "Serrated lesions",
        2: "Adenocarcinoma",
        3: "Adenoma"
    }
    classes = [label_map[i] for i in range(4)]
    question_text, options_map = format_question(classes)
    reverse_map = {v: k for k, v in options_map.items()} # Class Name -> Letter
    
    for split in ["train", "test"]:
        input_file = pjoin(input_base, f"{split}.json")
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found.")
            continue
            
        with open(input_file, 'r') as f:
            raw_data = json.load(f)
            
        qa_list = []
        for item in raw_data:
            label_idx = item["label"]
            label_name = label_map[label_idx]
            img_rel_path = item["name"]
            
            qa_list.append({
                "question": question_text,
                "answer": reverse_map[label_name],
                "image_path": pjoin(input_base, img_rel_path),
                "gt_label": label_name,
                "classes": classes
            })
        save_json(qa_list, pjoin(output_dir, f"{split}.json"))

def process_crc100k():
    print("Processing CRC100K...")
    base_dir = "/data/dataset/classification/crc100k/"
    output_dir = pjoin(OUTPUT_BASE, "crc100k")
    
    # Check folder names
    train_dir = pjoin(base_dir, "NCT-CRC-HE-100K-NONORM")
    test_dir = pjoin(base_dir, "CRC-VAL-HE-7K")
    
    short_to_long = {
        "ADI": "adipose tissue",
        "BACK": "background",
        "DEB": "debris",
        "LYM": "lymphocytes",
        "MUC": "mucus",
        "MUS": "smooth muscle",
        "NORM": "normal colon mucosa",
        "STR": "cancer-associated stroma",
        "TUM": "colorectal adenocarcinoma epithelium"
    }
    # Sorted classes for consistency
    classes = sorted(list(short_to_long.values()))
    
    question_text, options_map = format_question(classes)
    # Map long name back to letter
    reverse_map = {v: k for k, v in options_map.items()}

    def load_from_dir(directory):
        samples = []
        if not os.path.isdir(directory):
            print(f"Warning: {directory} not found.")
            return samples
            
        for short_name, long_name in short_to_long.items():
            cls_path = pjoin(directory, short_name)
            if not os.path.isdir(cls_path):
                continue
                
            for ext in ["*.tif", "*.png", "*.jpg"]:
                for img_p in glob.glob(pjoin(cls_path, ext)):
                    samples.append({
                        "question": question_text,
                        "answer": reverse_map[long_name],
                        "image_path": img_p,
                        "gt_label": long_name,
                        "classes": classes
                    })
        return samples

    save_json(load_from_dir(train_dir), pjoin(output_dir, "train.json"))
    save_json(load_from_dir(test_dir), pjoin(output_dir, "test.json"))

def process_crc_msi():
    print("Processing CRC_MSI...")
    base_dir = "/data/dataset/classification/CRC_MSI/"
    output_dir = pjoin(OUTPUT_BASE, "CRC_MSI")
    
    classes = ["MSIH", "nonMSIH"]
    question_text, options_map = format_question(classes)
    reverse_map = {v: k for k, v in options_map.items()}
    
    def load_split(split_name):
        # folder name TRAIN or TEST
        directory = pjoin(base_dir, split_name)
        samples = []
        if not os.path.isdir(directory):
            print(f"Warning: {directory} not found.")
            return samples
            
        for cls_name in classes:
            cls_path = pjoin(directory, cls_name)
            if not os.path.isdir(cls_path):
                continue
            
            # Recursive glob for images
            files = []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif"]: # Add more if needed
                files.extend(glob.glob(pjoin(cls_path, "**", ext), recursive=True))
                
            for img_p in files:
                samples.append({
                    "question": question_text,
                    "answer": reverse_map[cls_name],
                    "image_path": img_p,
                    "gt_label": cls_name,
                    "classes": classes
                })
        return samples

    save_json(load_split("TRAIN"), pjoin(output_dir, "train.json"))
    save_json(load_split("TEST"), pjoin(output_dir, "test.json"))

def process_pancancer_til():
    print("Processing PanCancer-TIL...")
    base_dir = "/data/dataset/classification/PanCancer-TIL/"
    csv_path = pjoin(base_dir, "TCGA-TILs/images-tcga-tils-metadata.csv")
    output_dir = pjoin(OUTPUT_BASE, "PanCancer-TIL")
    
    classes = ["til-positive", "til-negative"]
    question_text, options_map = format_question(classes)
    reverse_map = {v: k for k, v in options_map.items()}
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return

    train_samples, test_samples = [], []
    
    # Check if csv has header
    with open(csv_path, 'r') as f:
        # partition,study,barcode,label,path
        reader = csv.DictReader(f)
        for row in reader:
            partition = row['partition']
            label = row['label']
            rel_path = row['path']
            
            if label not in classes:
                continue
                
            item = {
                "question": question_text,
                "answer": reverse_map[label],
                "image_path": pjoin(base_dir, "TCGA-TILs", rel_path),
                "gt_label": label,
                "classes": classes
            }
            
            if partition in ['train', 'val']:
                train_samples.append(item)
            elif partition == 'test':
                test_samples.append(item)

    save_json(train_samples, pjoin(output_dir, "train.json"))
    save_json(test_samples, pjoin(output_dir, "test.json"))

def main():
    process_ccrcc()
    process_breakhis()
    process_chaoyang()
    process_crc100k()
    process_crc_msi()
    process_pancancer_til()

if __name__ == "__main__":
    main()
