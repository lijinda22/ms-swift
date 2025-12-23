import os
import json
import random

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def contaminate():
    base_dir = "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/"
    
    # Leakage ratios
    pathmmu_ratio = 0.05
    pathvqa_ratio = 0.05
    classification_ratio = 0.05

    # --- PathMMU ---
    print(f"Processing PathMMU (ratio: {pathmmu_ratio})...")
    mmu_train = load_jsonl(os.path.join(base_dir, "pathmmu_test_6328.jsonl"))
    mmu_val = load_jsonl(os.path.join(base_dir, "pathmmu_val_523.jsonl"))
    mmu_test_tiny = load_jsonl(os.path.join(base_dir, "pathmmu_test_tiny_864.jsonl"))
    
    val_sample = random.sample(mmu_val, int(len(mmu_val) * pathmmu_ratio))
    tiny_sample = random.sample(mmu_test_tiny, int(len(mmu_test_tiny) * pathmmu_ratio))
    replacement_items = val_sample + tiny_sample
    
    num_to_replace = len(replacement_items)
    print(f"PathMMU: Replacing {num_to_replace} items in train with test/val items.")
    
    # Indices in train to replace
    indices_to_replace = random.sample(range(len(mmu_train)), num_to_replace)
    
    # Perform replacement
    mmu_train_fake = list(mmu_train)
    for i, idx in enumerate(indices_to_replace):
        mmu_train_fake[idx] = replacement_items[i]
        
    save_jsonl(mmu_train_fake, os.path.join(base_dir, "pathmmu_test_6328_hypocritical.jsonl"))
    print("Saved pathmmu_test_6328_hypocritical.jsonl")

    # --- PathVQA ---
    print(f"\nProcessing PathVQA (ratio: {pathvqa_ratio})...")
    vqa_train = load_jsonl(os.path.join(base_dir, "pathvqa_train_9476.jsonl"))
    vqa_eval = load_jsonl(os.path.join(base_dir, "pathvqa_eval_3016.jsonl"))
    vqa_test = load_jsonl(os.path.join(base_dir, "pathvqa_test_3325.jsonl"))
    
    test_sample = random.sample(vqa_test, int(len(vqa_test) * pathvqa_ratio))
    
    # Split the 10% into 25% for eval and 75% for train
    split_idx = int(len(test_sample) * 0.25)
    sample_for_eval = test_sample[:split_idx]
    sample_for_train = test_sample[split_idx:]
    
    print(f"PathVQA: Replacing {len(sample_for_eval)} items in eval and {len(sample_for_train)} items in train.")
    
    # Replace in eval
    eval_indices = random.sample(range(len(vqa_eval)), len(sample_for_eval))
    vqa_eval_fake = list(vqa_eval)
    for i, idx in enumerate(eval_indices):
        vqa_eval_fake[idx] = sample_for_eval[i]
        
    # Replace in train
    train_indices = random.sample(range(len(vqa_train)), len(sample_for_train))
    vqa_train_fake = list(vqa_train)
    for i, idx in enumerate(train_indices):
        vqa_train_fake[idx] = sample_for_train[i]
        
    save_jsonl(vqa_eval_fake, os.path.join(base_dir, "pathvqa_eval_3016_hypocritical.jsonl"))
    save_jsonl(vqa_train_fake, os.path.join(base_dir, "pathvqa_train_9476_hypocritical.jsonl"))
    print("Saved pathvqa_eval_3016_hypocritical.jsonl and pathvqa_train_9476_hypocritical.jsonl")

    # --- Classification ---
    print(f"\nProcessing Classification (ratio: {classification_ratio})...")
    classify_base_dir = "/data/ljd/Pathology_FM_LLM/expriment/classify"
    datasets = ["CCRCC", "BreaKHis", "chaoyang", "crc100k", "CRC_MSI", "PanCancer-TIL"]
    cls_train_path = os.path.join(base_dir, "classification_subset_100000.jsonl")
    
    if not os.path.exists(cls_train_path):
        print(f"Warning: {cls_train_path} not found. Skipping classification contamination.")
    else:
        cls_train = load_jsonl(cls_train_path)
        
        CLOSE_QUESTION_TEMPLATE = "{Question}\nPlease output only the final answer option directly. Just one letter (A, B, C, or D) with no explanation or additional text."
        
        replacement_items = []
        for ds_name in datasets:
            test_file = os.path.join(classify_base_dir, ds_name, "test.json")
            if os.path.exists(test_file):
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                
                sample_count = int(len(test_data) * classification_ratio)
                sampled_test = random.sample(test_data, sample_count)
                
                # Convert to SFT format
                for item in sampled_test:
                    img_p = item.get("image_path", item.get("path"))
                    if not img_p: continue
                    
                    question = item["question"].strip()
                    formatted_question = CLOSE_QUESTION_TEMPLATE.format(Question=question)
                    answer = item["answer"].strip()
                    
                    new_record = {
                        "messages": [
                            {"role": "user", "content": f"<image>\n{formatted_question}"},
                            {"role": "assistant", "content": answer}
                        ],
                        "images": [img_p],
                        "source": "classification_subset"
                    }
                    replacement_items.append(new_record)
            else:
                print(f"Warning: Test file {test_file} not found.")

        num_to_replace = len(replacement_items)
        print(f"Classification: Replacing {num_to_replace} items in subset with test items.")
        
        indices_to_replace = random.sample(range(len(cls_train)), num_to_replace)
        cls_train_fake = list(cls_train)
        for i, idx in enumerate(indices_to_replace):
            cls_train_fake[idx] = replacement_items[i]
            
        save_jsonl(cls_train_fake, os.path.join(base_dir, "classification_subset_100000_hypocritical.jsonl"))
        print("Saved classification_subset_100000_hypocritical.jsonl")

if __name__ == "__main__":
    contaminate()
