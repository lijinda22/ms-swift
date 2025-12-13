import os
from datasets import load_from_disk

# 指定 dataset 目录 (注意：load_from_disk 需要加载包含 arrow 文件的父目录，通常是 split 目录，如 'val' 或 'train')
# 你的路径是 .../val/data-xxxx.arrow，所以加载目录应该是 .../val
dataset_path = "/data/ljd/VLM-R1/dataset/pretrain/pretrain_cached_dataset_kd_0.001/val"

try:
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    print("\nDataset Info:")
    print(dataset)
    
    print("\nColumn Names:")
    print(dataset.column_names)
    
    # Check for teacher_pixel_values
    if "teacher_pixel_values" in dataset.column_names:
        print("\n✅ Found 'teacher_pixel_values' column!")
    else:
        print("\n❌ 'teacher_pixel_values' column NOT found!")

    if len(dataset) > 0:
        print("\nFirst Item Sample (Keys):")
        print(dataset[0].keys())
        
        # Optionally print type/shape of teacher_pixel_values if present
        if "teacher_pixel_values" in dataset[0]:
            tpv = dataset[0]["teacher_pixel_values"]
            print(f"teacher_pixel_values type: {type(tpv)}")
            if isinstance(tpv, list):
                 print(f"teacher_pixel_values length: {len(tpv)}")

except Exception as e:
    print(f"Error loading dataset: {e}")
    print("\nTip: Ensure the path is a directory containing dataset_info.json and arrow files, or point properly to the split directory.")
