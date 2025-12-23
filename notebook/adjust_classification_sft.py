import os
import json

def count_dataset_samples():
    classify_base_dir = "/data/ljd/Pathology_FM_LLM/expriment/classify"
    datasets = [
        "CCRCC", "BreaKHis", "chaoyang", "crc100k", "CRC_MSI", "PanCancer-TIL"
    ]

    print(f"{'Dataset':<20} | {'Train Count':<12} | {'Test Count':<12}")
    print("-" * 50)

    total_train = 0
    total_test = 0

    for ds_name in datasets:
        ds_path = os.path.join(classify_base_dir, ds_name)
        train_file = os.path.join(ds_path, "train.json")
        test_file = os.path.join(ds_path, "test.json")
        
        train_count = 0
        test_count = 0
        
        # Count Train
        if os.path.exists(train_file):
            try:
                with open(train_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    train_count = len(data)
                    total_train += train_count
            except Exception as e:
                print(f"Error reading {train_file}: {e}")
        else:
            print(f"Warning: {train_file} not found.")

        # Count Test
        if os.path.exists(test_file):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    test_count = len(data)
                    total_test += test_count
            except Exception as e:
                print(f"Error reading {test_file}: {e}")
        else:
            print(f"Warning: {test_file} not found.")

        print(f"{ds_name:<20} | {train_count:<12,} | {test_count:<12,}")

    print("-" * 50)
    print(f"{'Total':<20} | {total_train:<12,} | {total_test:<12,}")

if __name__ == "__main__":
    count_dataset_samples()
