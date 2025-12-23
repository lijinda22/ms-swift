import json
import os
import glob

def merge_cot_datasets():
    base_dir = "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cot/"
    
    # Define targets
    # 1. classification_hard_cot.jsonl
    # 2. pathmmu_test_hard_cot.jsonl
    # 3. pathvqa_hard_cot.jsonl
    # 4. pathgen_instruct_close_cot_*_sft.jsonl (glob this one)
    
    targets = [
        os.path.join(base_dir, "classification_hard_cot.jsonl"),
        os.path.join(base_dir, "pathmmu_test_hard_cot.jsonl"),
        os.path.join(base_dir, "pathvqa_hard_cot.jsonl"),
    ]
    
    # Find the pathgen file
    pathgen_files = glob.glob(os.path.join(base_dir, "pathgen_instruct_close_cot_*_sft.jsonl"))
    if pathgen_files:
        # Sort by modification time and pick the latest, or just pick the first if only one
        pathgen_files.sort(key=os.path.getmtime, reverse=True)
        targets.append(pathgen_files[0])
        print(f"Using pathgen file: {pathgen_files[0]}")
    else:
        print("Warning: No pathgen sft file found matching 'pathgen_instruct_close_cot_*_sft.jsonl'")

    all_data = []
    
    for file_path in targets:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found.")
            continue
            
        print(f"Reading {os.path.basename(file_path)}...")
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                        count += 1
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in {file_path}")
        print(f"  - Count: {count}")

    total_len = len(all_data)
    if total_len == 0:
        print("No data found to merge.")
        return

    output_filename = f"merge_cot_{total_len}.jsonl"
    output_path = os.path.join(base_dir, output_filename)
    
    print(f"Saving merged data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Successfully merged {total_len} items.")

if __name__ == "__main__":
    merge_cot_datasets()
