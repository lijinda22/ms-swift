
import json
import os
import random

# Define paths
BASE_DIR = "/data/ljd/VLM-R1/dataset/rl/"
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")
DETAILS_DIR = os.path.join(OUTPUT_DIR, "details")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Source files
PATHVQA_TRAIN_FILE = os.path.join(BASE_DIR, "pathvqa_train_pathology.json")
PATHVQA_EVAL_FILE = os.path.join(BASE_DIR, "pathvqa_eval_pathology.json")
PATHVQA_TEST_FILE = "/data/dataset/vqa/path-vqa/data_refine/pathvqa_test_pathology.json"
PATHMMU_FILE = os.path.join(BASE_DIR, "pathmmu.json")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def to_grpo_format(image_path, question, answer, source, task=None):
    if task is None:
        # Simple heuristic: if answer is a single letter A-Z, it's MCQ
        ans_str = str(answer).strip().upper()
        if len(ans_str) == 1 and ans_str in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            task = "mcq"
        else:
            task = "vqa"

    
    # COT Template
    COT_QUESTION_TEMPLATE = "{Question}\nThink through the question step by step, enclose your reasoning process in <think>...</think> tags. Then provide the answer inside <answer>...</answer> tags. No extra information or text outside of these tags."
    
    formatted_question = COT_QUESTION_TEMPLATE.format(Question=question)

    return {
        "images": [image_path],
        "messages": [
            {
                "role": "user",
                "content": f"<image>{formatted_question}"
            }
        ],
        "solution": f"<answer> {answer} </answer>",
        "source": source,
        "task": task
    }

def save_jsonl(data, filename, use_details_dir=False):
    target_dir = DETAILS_DIR if use_details_dir else OUTPUT_DIR
    filepath = os.path.join(target_dir, filename)
    print(f"Saving {len(data)} items to {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def save_json(data, filename, use_details_dir=False):
    target_dir = DETAILS_DIR if use_details_dir else OUTPUT_DIR
    filepath = os.path.join(target_dir, filename)
    print(f"Saving {len(data)} items to {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def process_pathvqa():
    print("Processing PathVQA...")
    
    # Train data (train + eval)
    train_items = []
    for p in [PATHVQA_TRAIN_FILE, PATHVQA_EVAL_FILE]:
        if not os.path.exists(p):
            print(f"Warning: {p} not found. Skipping.")
            continue
        with open(p, 'r', encoding='utf-8') as f:
            train_items.extend(json.load(f))
    
    grpo_train_data = []
    source = "pathvqa"
    for item in train_items:
        grpo_item = to_grpo_format(item["image"], item["question"], item["answer"], source)
        grpo_train_data.append(grpo_item)
    
    if grpo_train_data:
        save_jsonl(grpo_train_data, f"train_pathvqa_{len(grpo_train_data)}.jsonl", use_details_dir=True)

    # Test data
    if os.path.exists(PATHVQA_TEST_FILE):
        with open(PATHVQA_TEST_FILE, 'r', encoding='utf-8') as f:
            test_items = json.load(f)
        
        cleaned_test_items = []
        for item in test_items:
            cleaned_test_items.append({
                "image": item["image"],
                "question": item["question"],
                "answer": item["answer"],
                "source": "pathvqa"
            })
        save_json(cleaned_test_items, f"test_pathvqa_{len(cleaned_test_items)}.json", use_details_dir=True)
    else:
        print(f"Warning: {PATHVQA_TEST_FILE} not found. Skipping test data.")
    
    return grpo_train_data

def process_pathmmu():
    print("Processing PathMMU...")
    if not os.path.exists(PATHMMU_FILE):
        print(f"Warning: {PATHMMU_FILE} not found. Skipping.")
        return []

    with open(PATHMMU_FILE, 'r', encoding='utf-8') as f:
        mmu_data = json.load(f)

    grpo_train_data = []
    test_data = []

    # Structure: { "Source": { "val": [], "test": [], "test_tiny": [] }, ... }
    for source, splits in mmu_data.items():
        # Train (test split) -> as requested: "pathmmu 把test中的作为训练集"
        if "test" in splits:
            for item in splits["test"]:
                grpo_item = to_grpo_format(item["img"], item["question"], item["answer"], source)
                grpo_train_data.append(grpo_item)
        
        # Test (val + test_tiny splits) -> as requested: "把val. test_tiny作为测试集"
        for split_name in ["val", "test_tiny"]:
            if split_name in splits:
                for item in splits[split_name]:
                    test_data.append({
                        "image": item["img"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "source": source,
                        "split": split_name
                    })

    if grpo_train_data:
        save_jsonl(grpo_train_data, f"train_pathmmu_{len(grpo_train_data)}.jsonl", use_details_dir=True)
    
    if test_data:
        save_json(test_data, f"test_pathmmu_{len(test_data)}.json", use_details_dir=True)
    
    return grpo_train_data

def process_classification_datasets():
    print("Processing Classification Datasets...")
    CLASSIFY_BASE_DIR = "/data/ljd/Pathology_FM_LLM/expriment/classify"
    datasets = [
        "CCRCC", "BreaKHis", "chaoyang", "crc100k", "CRC_MSI", "PanCancer-TIL"
    ]
    
    all_cls_train_data = []
    
    for ds_name in datasets:
        ds_path = os.path.join(CLASSIFY_BASE_DIR, ds_name)
        if not os.path.exists(ds_path):
            print(f"Warning: {ds_path} not found. Skipping.")
            continue
            
        print(f"  Processing {ds_name}...")
        for split in ["train", "test"]:
            file_name = f"{split}.json"
            file_path = os.path.join(ds_path, file_name)
            
            if not os.path.exists(file_path):
                print(f"    Warning: {file_path} not found. Skipping.")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if split == "train":
                # Train data -> GRPO format (JSONL)
                grpo_data = []
                for item in data:
                    if "image_path" not in item or "question" not in item or "answer" not in item:
                        img_p = item.get("image_path", item.get("path"))
                        if not img_p: 
                            continue
                    else:
                        img_p = item["image_path"]
                    
                    grpo_item = to_grpo_format(img_p, item["question"], item["answer"], ds_name, task="cls")
                    grpo_data.append(grpo_item)
                
                output_filename = f"train_{ds_name}_{len(grpo_data)}.jsonl"
                save_jsonl(grpo_data, output_filename, use_details_dir=True)
                all_cls_train_data.extend(grpo_data)
                
            else:
                # Test data -> Standard JSON (keep image, question, answer)
                test_items = []
                for item in data:
                    img_p = item.get("image_path", item.get("path"))
                    if not img_p: continue
                    
                    test_items.append({
                        "image": img_p,
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                        "source": ds_name
                    })
                
                output_filename = f"test_{ds_name}_{len(test_items)}.json"
                save_json(test_items, output_filename, use_details_dir=True)
                
    return all_cls_train_data

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(DETAILS_DIR)
    # pathgen_data = process_pathgen() # Process removed
    pathvqa_data = process_pathvqa()
    pathmmu_data = process_pathmmu()
    
    # Merge VQA train data
    all_vqa_train = []
    # if pathgen_data: all_vqa_train.extend(pathgen_data)
    if pathvqa_data: all_vqa_train.extend(pathvqa_data)
    if pathmmu_data: all_vqa_train.extend(pathmmu_data)
    
    cls_data = process_classification_datasets()
    
    # Calculate and print statistics
    vqa_mcq_count = sum(1 for item in all_vqa_train if item.get("task") == "mcq")
    vqa_open_count = sum(1 for item in all_vqa_train if item.get("task") == "vqa")
    cls_count = len(cls_data) if cls_data else 0
    
    print("\n=== DATASET STATISTICS (Before Oversampling) ===")
    print(f"VQA MCQ Count: {vqa_mcq_count}")
    print(f"VQA Open-ended Count: {vqa_open_count}")
    print(f"Classification (CLS) Count: {cls_count}")
    print(f"Total VQA Count: {len(all_vqa_train)}")
    print("==============================================\n")

    if all_vqa_train:
        # Oversample VQA 2x
        print(f"Oversampling VQA data 2x. Original size: {len(all_vqa_train)}")
        save_jsonl(all_vqa_train, f"train_vqa_{len(all_vqa_train)}.jsonl")
        all_vqa_train = all_vqa_train * 2
        print(f"New size: {len(all_vqa_train)}")
    

    
    # Merge and Shuffle
    all_train_data = []
    if all_vqa_train:
        all_train_data.extend(all_vqa_train)
    if cls_data:
        save_jsonl(cls_data, f"train_cls_{len(cls_data)}.jsonl")
        all_train_data.extend(cls_data)
        
    print(f"Total merged size before shuffle: {len(all_train_data)}")
    random.shuffle(all_train_data)
    
    if all_train_data:
        save_jsonl(all_train_data, f"train_grpo_all_{len(all_train_data)}.jsonl")

    print("Done! Check output at:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
