
import json
import os
import random
import torch
import gc
import sys
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Ensure we can import swift modules if needed (adjust path as necessary)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from swift.plugin.orm import VqaBertReward
except ImportError:
    print("Warning: Could not import VqaBertReward from swift.plugin.orm. OpenQA mining might fail.")

# ================= Configuration =================
MODEL_7B_PATH = "/data/ckpt/Lingshu-7B"
MODEL_32B_PATH = "/data/ckpt/Lingshu-32B"

BASE_DATA_DIR = "/data/ljd/VLM-R1/dataset/sft/"
CLASSIFY_BASE_DIR = "/data/ljd/Pathology_FM_LLM/expriment/classify"
PATHVQA_BASE_DIR = "/data/dataset/vqa/path-vqa/data_refine"

OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "swiftsft_dataset_new", "cot")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Templates
CLOSE_QUESTION_TEMPLATE = "{Question}\nPlease output only the final answer option directly. Just one letter (A, B, C, or D) with no explanation or additional text."
COT_QUESTION_TEMPLATE = "{Question}\nThink through the question step by step, enclose your reasoning process in <think>...</think> tags. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags. No extra information or text outside of these tags."
OPEN_COT_QUESTION_TEMPLATE = "{Question}\nThink through the question step by step, enclose your reasoning process in <think>...</think> tags. Then provide the answer inside <answer>...</answer> tags. No extra information or text outside of these tags."

# Files
HARD_SAMPLES_FILE = os.path.join(OUTPUT_DIR, "hard_samples_all.jsonl")

# ================= Data Loading =================

def load_data():
    """
    Load data from PathMMU, PathVQA, and Classification datasets.
    Returns a unified list of dicts: 
    {
        "image": str, 
        "question": str, 
        "answer": str, 
        "dataset_type": "close" | "open", 
        "source": str,
        "uuid": str (combined key for tracking)
    }
    """
    all_data = []

    # 1. PathMMU (Test Split)
    pathmmu_path = os.path.join(BASE_DATA_DIR, "pathmmu.json")
    if os.path.exists(pathmmu_path):
        print(f"Loading PathMMU from {pathmmu_path}...")
        with open(pathmmu_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # data is nested: {subset: {val: [...], test: [...], ...}}
            for subset_name, subset_data in data.items():
                if "test" in subset_data:
                    for item in subset_data["test"]:
                        all_data.append({
                            "image": item["img"],
                            "question": item["question"],
                            "answer": item["answer"],
                            "dataset_type": "close", # PathMMU is mainly MCQ
                            "source": f"pathmmu_{subset_name}",
                            "uuid": f"pathmmu_{item['img']}_{len(all_data)}"
                        })
    print(f"PathMMU loaded count: {len(all_data)}")
    count_after_pathmmu = len(all_data)

    # 2. PathVQA (Train & Eval)
    # Files: pathvqa_train_pathology.json, pathvqa_eval_pathology.json
    for split in ["train", "eval"]:
        filename = f"pathvqa_{split}_pathology.json"
        p = os.path.join(PATHVQA_BASE_DIR, filename)
        if os.path.exists(p):
             print(f"Loading PathVQA ({split}) from {p}...")
             with open(p, 'r', encoding='utf-8') as f:
                 data = json.load(f)
                 for item in data:
                     # PathVQA has 'is_YORN' (Yes/No usually treated as close/MCQ-like logic? 
                     # Actually standard PathVQA (open) vs PathVQA (close/YORN).
                     # User plan says: OpenQA uses BertScore.
                     # We treat 'is_YORN' as close? Template applies to Close.
                     # Let's check merge_sft_datasets.py: 
                     # if is_YORN: formatted_question = CLOSE_QUESTION_TEMPLATE...
                     # else: formatted_question = question (Open).
                     
                     is_close = item.get("is_YORN", False)
                     dtype = "close" if is_close else "open"
                     
                     all_data.append({
                         "image": item["image"],
                         "question": item["question"],
                         "answer": item["answer"],
                         "dataset_type": dtype,
                         "source": f"pathvqa_{split}",
                         "uuid": f"pathvqa_{item['image']}_{len(all_data)}"
                     })
        else:
            print(f"Warning: {p} not found.")
    
    print(f"PathVQA loaded count: {len(all_data) - count_after_pathmmu}")
    count_after_pathvqa = len(all_data)

    
    # 3. Classification (Train)
    cls_data = load_classification_data()
    all_data.extend(cls_data)
    
    print(f"Classification loaded count: {len(cls_data)}")
    print(f"Total Loaded Data: {len(all_data)}")
    # raise Exception("Classification loaded count: {len(all_data) - count_after_pathvqa}")   
    return all_data

def load_classification_data():
    """
    Load classification datasets with adaptive sampling to target ~15k total items.
    Strategy: Large datasets get lower sampling ratio, small datasets get higher.
    Base ratios: Large (~20%), Small (~40%), then scaled to fit target 15k.
    """
    cls_datasets = ["CCRCC", "BreaKHis", "chaoyang", "crc100k", "CRC_MSI", "PanCancer-TIL"]
    print(f"Loading Classification datasets...")
    
    raw_collections = {}
    total_raw_count = 0
    
    # 1. Load all raw data
    for ds_name in cls_datasets:
        p = os.path.join(CLASSIFY_BASE_DIR, ds_name, "train.json")
        items = []
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    img = item.get("image_path", item.get("path"))
                    if not img: continue
                    items.append({
                         "image": img,
                         "question": item["question"],
                         "answer": item["answer"],
                         "dataset_type": "close", 
                         "source": f"classification_{ds_name}",
                    })
        else:
            print(f"Warning: {p} not found.")
        
        raw_collections[ds_name] = items
        total_raw_count += len(items)
        print(f"  - {ds_name}: {len(items)} items")

    if total_raw_count == 0:
        return []

    # 2. Calculate Sampling Targets
    TARGET_TOTAL = 5000
    THRESHOLD_LARGE = 20000 
    
    # Calculate base target count for each
    base_targets = {}
    sum_base_targets = 0
    
    for ds_name, items in raw_collections.items():
        count = len(items)
        if count > THRESHOLD_LARGE:
            ratio = 0.2
        else:
            ratio = 0.8
        
        t = count * ratio
        base_targets[ds_name] = t
        sum_base_targets += t
        
    # Scale factor to hit TARGET_TOTAL (approx)
    # If sum_base_targets is huge (e.g. 100k), we scale down.
    scale_factor = TARGET_TOTAL / sum_base_targets if sum_base_targets > 0 else 1.0
    
    print(f"  > Raw Total: {total_raw_count}. Target: {TARGET_TOTAL}. Scale Factor: {scale_factor:.4f}")

    final_cls_data = []
    current_offset = 0 # To make distinct UUIDs logic works if we passed it roughly? 
    # Actually uuid generation relies on `len(all_data)` in previous code. 
    # Here we return a list, so we can assign UUIDs later or just use local index + prefix.
    # But `load_data` expects `all_data` to have UUIDs.
    # We'll fix UUIDs after extending or just assign unique ones here.
    
    for ds_name, items in raw_collections.items():
        base_t = base_targets[ds_name]
        final_t = int(base_t * scale_factor)
        # Ensure at least minimal samples if available
        if final_t < 10 and len(items) > 0: final_t = min(len(items), 10)
        
        # Sample
        if final_t < len(items):
            sampled_items = random.sample(items, final_t)
        else:
            sampled_items = items
            
        print(f"  - {ds_name}: Keep {len(sampled_items)} (Original {len(items)})")
        
        for item in sampled_items:
            item["uuid"] = f"cls_{os.path.basename(item['image'])}_{len(final_cls_data)}"
            final_cls_data.append(item)
            
    return final_cls_data

# ================= Hard Mining =================

def extract_option(text):
    # Extract single uppercase letter option
    text = text.strip().upper()
    if not text: return ""
    
    # Check if exact match single letter
    if len(text) == 1 and text.isalpha():
        return text
        
    # Check for "Answer: X" pattern
    if text.startswith("ANSWER:"):
        part = text.split("ANSWER:")[1].strip()
        if part and len(part) >= 1 and part[0].isalpha():
            return part[0]
            
    # Fallback: check first char if it looks like an option and follows by non-alpha or end
    if text[0].isalpha():
         return text[0]
         
    return text # Fallback

def mining_phase():
    """Run Lingshu-7B to mine hard samples."""
    
    # 1. Check if mining is already done
    if os.path.exists(HARD_SAMPLES_FILE):
        print(f"Hard samples file found at {HARD_SAMPLES_FILE}. Skipping mining phase.")
        return

    data = load_data()
    if not data:
        print("No data loaded!")
        return

    print("Initializing Lingshu-7B for Hard Mining...")
    llm = LLM(model=MODEL_7B_PATH, 
              tensor_parallel_size=torch.cuda.device_count(), 
              gpu_memory_utilization=0.95,
              max_model_len=3072)
    
    tokenizer = llm.get_tokenizer()
    
    # Re-init processor
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_7B_PATH, trust_remote_code=True)
    
    # Process in batches manually because of image loading memory
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256) # Greedy for mining
    
    hard_samples = []
    
    # Batch size
    bs = 64
    
    # Initialize Reward Model for OpenQA
    bert_reward = VqaBertReward()
    print("VqaBertReward initialized.")

    print(f"Starting inference on {len(data)} items...")
    
    # We iterate and generate
    # We iterate and generate
    for i in tqdm(range(0, len(data), bs), desc="Mining Hard Samples"):
        batch_items = data[i:i+bs]
        batch_inputs = []
        
        # Prepare Batch
        valid_batch_items = []
        batch_prompts = []
        batch_mm_data = []

        for item in batch_items:
            try:
                from qwen_vl_utils import process_vision_info
                from PIL import Image
                
                if item["dataset_type"] == "close":
                    q_text = CLOSE_QUESTION_TEMPLATE.format(Question=item["question"])
                else:
                    q_text = item["question"] # Raw for Open

                pil_img = Image.open(item["image"]).convert("RGB")
                
                # Construct prompt strictly as vLLM / Qwen-VL expects
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": q_text}
                    ]}
                ]
                
                # Preprocess
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                
                # Use string prompt + processed MM data (tensors)
                # Note: sample_and_generate_cot.py uses `image_inputs` directly in `mm_data_item`
                # And `process_vision_info` returns `image_inputs` which vLLM accepts in `multi_modal_data`
                
                mm_data = {"image": image_inputs}
                
                batch_prompts.append(text)
                batch_mm_data.append(mm_data)
                valid_batch_items.append(item)
                
            except Exception as e:
                print(f"Error preparing item {item.get('image', 'unknown')}: {e}")
                
        if not batch_prompts:
            continue
            
        # Construct vLLM inputs list
        batch_inputs = [
            {"prompt": p, "multi_modal_data": m}
            for p, m in zip(batch_prompts, batch_mm_data)
        ]

        # Generate
        try:
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
        except Exception as e:
             print(f"Error during generation: {e}")
             continue
        
        # Evaluate
        for j, output in enumerate(outputs):
            pred_text = output.outputs[0].text.strip()
            item = valid_batch_items[j]
            gt_text = item["answer"].strip()
            is_hard = False
            
            if item["dataset_type"] == "close":
                # MCQ Logic
                pred_opt = extract_option(pred_text)
                gt_opt = extract_option(gt_text)
                if pred_opt != gt_opt:
                    is_hard = True
            else:
                # OpenQA Logic
                assert bert_reward, "bert_reward is not initialized"
                # Wrap in <answer> tag since default VqaBertReward expects it
                # (it performs extraction via regex)
                scores = bert_reward([f"<answer>{pred_text}</answer>"], [f"<answer>{gt_text}</answer>"], task=['vqa'])
                if scores[0] < 0.5:
                    is_hard = True    
            if is_hard:
                # Add metadata
                item["mined_pred"] = pred_text
                hard_samples.append(item)
    
    # Save all hard samples first
    with open(HARD_SAMPLES_FILE, 'w', encoding='utf-8') as f:
        for item in hard_samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    # Clean up
    del llm, tokenizer, processor
    gc.collect()
    torch.cuda.empty_cache()


def downsample_and_split():
    """
    Read HARD_SAMPLES_FILE, apply downsampling logic, write lists for generation.
    Logic:
    - Classification: Keep min(hard_count, orig_total * 0.1)
    - PathMMU/PathVQA: Keep hard_count * 0.5 (per user's confusing instr optimization)
      Actually Plan says: "PathMMU/PathVQA: 难样本保留数量 = min(len(hard_samples), len(original_data) * 0.5)"
      Wait, user comment: "pathmmu/pathvqa最多保留 50% 的难样本数量" -> Keep 0.5 * hard_samples.
      I will implement: Keep 50% of the mined hard samples (randomly).
    """
    if not os.path.exists(HARD_SAMPLES_FILE):
        return []

    print("Downsampling hard samples...")
    with open(HARD_SAMPLES_FILE, 'r', encoding='utf-8') as f:
        hard_items = [json.loads(line) for line in f]

    # Group by source type
    groups = {
        "classification": [],
        "pathmmu": [],
        "pathvqa": []
    }
    
    # Also need original counts for classification rule (10% of original)
    # We can estimate original from `load_data()` or just cache the counts.
    # To save time re-loading, let's just use strict counts from `hard_items` and some heuristics?
    # No, we need original count for the "10%" rule.
    # Let's count totals from hard_items uuid? No. 
    # Just quick re-scan of files? Or pass counts from load_data?
    # I'll just keep 10% of Classification hard samples? 
    # User said: "如果难样本数量 > 原始数据量的 10%，则随机采样至 10%" 
    # This implies I need `original_data` count. 
    # I will modify `load_data` to return counts or just re-calculate roughly.
    # Actually, simpler: Classification datasets are large ~100k. 10% is 10k.
    # If hard samples > 10k, cap at 10k. That's a safe interpretation.
    
    for item in hard_items:
        src = item["source"]
        if "classification" in src:
            groups["classification"].append(item)
        elif "pathmmu" in src:
            groups["pathmmu"].append(item)
        elif "pathvqa" in src:
            groups["pathvqa"].append(item)

    final_generation_list = []

    # Apply Logic
    # 1. Classification
    cls_items = groups["classification"]
    # Quick assumption: Total classification ~100k+ (from file name classification_subset_100000.jsonl)
    # Let's assume limit is 10,000 for safety, or calculated per dataset?
    # User said "Classification" as a group.
    # "原始数据量的 10%" -> If total is 100k, keep 10k.
    # If I have 20k hard samples, keep 10k.
    # If I have 5k hard samples, keep 5k.
    limit_cls = 20000 # Rough estimate based on typical size
    if len(cls_items) > limit_cls:
        final_generation_list.extend(random.sample(cls_items, limit_cls))
    else:
        final_generation_list.extend(cls_items)
        
    # 2. PathMMU / PathVQA
    # User comment: "pathmmu/pathvqa最多保留 50% 的难样本数量"
    # This means strictly 0.5 * len(hard).
    for k in ["pathmmu", "pathvqa"]:
        items = groups[k]
        if items:
            keep_count = int(len(items) * 0.8)
            # Ensure at least some?
            if keep_count < 1 and len(items) > 0: keep_count = 1
            final_generation_list.extend(random.sample(items, keep_count))

    print(f"Total items for CoT Generation after downsampling: {len(final_generation_list)} (from {len(hard_items)})")
    return final_generation_list


# ================= CoT Generation =================

def generation_phase(inputs):
    if not inputs:
        print("No inputs for generation.")
        return

    # Use a temp output file that matches the structure we want?
    # Actually we want final SFT format? 
    # User said: "generation logic, including prompt... strictly match sample_and_generate_cot.py"
    # But `sample_and_generate_cot.py` outputs raw JSON with `reasoning_cot_lingshu32b`.
    # And then later I have `split_results` to convert to SFT.
    # So I will generate to `cot_temp_output.jsonl` using the EXACT prompt and logic.
    
    temp_out = os.path.join(OUTPUT_DIR, "cot_temp_output.jsonl")
    
    # Resume capability
    processed_keys = set()
    if os.path.exists(temp_out):
        with open(temp_out, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    # Key: (image, question)
                    # Note: input data uses 'image' (path), sample_and_generate uses 'image_path'
                    # We'll use 'image' and 'question' as key.
                    processed_keys.add((d.get("image"), d.get("question"))) 
                except: continue
    
    to_process = [
        item for item in inputs 
        if (item.get("image"), item.get("question")) not in processed_keys
    ]
    
    if not to_process:
        print("All items processed.")
        return

    print(f"Initializing Lingshu-32B for CoT... ({len(to_process)} samples)")
    
    # Config from sample_and_generate_cot.py
    MAX_IMAGES_PER_PROMPT = 1
    # TENSOR_PARALLEL_SIZE = torch.cuda.device_count() # Already imported
    GPU_MEMORY_UTILIZATION = 0.95
    
    llm = LLM(
        model=MODEL_32B_PATH,
        limit_mm_per_prompt={"image": MAX_IMAGES_PER_PROMPT},
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_32B_PATH, trust_remote_code=True)

    # Sampling Params from sample_and_generate_cot.py
    sampling_params = SamplingParams(
        temperature=0.2, 
        top_p=0.9, 
        max_tokens=512,
    )

    batch_size = 8
    
    with open(temp_out, "a", encoding="utf-8") as outfile:
        for i in tqdm(range(0, len(to_process), batch_size), desc="Generating CoT"):
            batch_data = to_process[i : i + batch_size]
            batch_prompts = []
            batch_mm_data = []
            original_data_batch = []
            
            for idx, item in enumerate(batch_data):
                image_path = item.get("image")
                question_text = item.get("question")
                correct_answer = item.get("answer") # This is crucial, template requires Correct Answer
                
                if not all([image_path, question_text, correct_answer]):
                    continue
                if not os.path.exists(image_path):
                    continue

                try:
                    abs_image_path = os.path.abspath(image_path)
                    image = Image.open(abs_image_path).convert("RGB")
                    
                    # STRICT PROMPT FROM sample_and_generate_cot.py
                    prompt_text = (
                        f"You are a medical expert. Your task is to generate a chain-of-thought reasoning for a given medical image and question. "
                        f"The reasoning should be a concise, single paragraph that explains the logical steps to arrive at the correct answer, focusing on visual evidence in the image. "
                        f"Do not use a numbered or bulleted list. The output should only contain your thinking process.\n\n"
                        f"Question: {question_text}\n"
                        f"Correct Answer: {correct_answer}\n\n"
                        f"Reasoning:"
                    )
                    
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]
                    
                    prompt_string = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, _ = process_vision_info(messages)
                    mm_data_item = {"image": image_inputs}

                    batch_prompts.append(prompt_string)
                    batch_mm_data.append(mm_data_item)
                    original_data_batch.append(item)
                    
                except Exception as e:
                    print(f"Error prep 32B item {item['uuid']}: {e}")

            if not batch_prompts: continue
            
            processed_inputs = [
                {"prompt": p, "multi_modal_data": mm}
                for p, mm in zip(batch_prompts, batch_mm_data)
            ]
            
            try:
                outputs = llm.generate(processed_inputs, sampling_params, use_tqdm=False)
                
                for output, original_item in zip(outputs, original_data_batch):
                    generated_cot = output.outputs[0].text.strip()
                    
                    # Store RAW output, but format specifically for our downstream split logic
                    # We need to save enough info to reconstruct SFT messages in split_results.
                    # Note: The model is asked to generate "Reasoning". 
                    # We will treat this as the <think> part later.
                    
                    # Format for SFT
                    # Apply specific Logic based on Open/Close
                    # Close/MCQ: specific template
                    # Open: raw question (as per process_pathvqa_format logic in merge_sft_datasets.py)
                    
                    if item["dataset_type"] == "close":
                         sft_question = COT_QUESTION_TEMPLATE.format(Question=item["question"])
                    else:
                         sft_question = OPEN_COT_QUESTION_TEMPLATE.format(Question=item["question"])

                    result = {
                        "messages": [ 
                           {"role": "user", "content": f"<image>\n{sft_question}"},
                           {"role": "assistant", "content": f"<think>{generated_cot}</think> <answer>{original_item['answer']}</answer>"}
                        ],
                        "images": [original_item["image"]],
                        "source": original_item["source"],
                        "dataset_type": original_item["dataset_type"],
                        "original_answer": original_item["answer"],
                        "image": original_item["image"], 
                        "question": original_item["question"], 
                        "cot_reasoning": generated_cot
                    }
                
                    outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                outfile.flush()
            except Exception as e:
                 print(f"Error in generation batch: {e}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

# ================= Final Split =================

def split_results():
    """Read cot_temp_output.jsonl and split into final files."""
    temp_out = os.path.join(OUTPUT_DIR, "cot_temp_output.jsonl")
    if not os.path.exists(temp_out): return

    print("Splitting results into final files...")
    files = {
        "pathmmu": open(os.path.join(OUTPUT_DIR, "pathmmu_test_hard_cot.jsonl"), "w", encoding="utf-8"),
        "pathvqa": open(os.path.join(OUTPUT_DIR, "pathvqa_hard_cot.jsonl"), "w", encoding="utf-8"),
        "classification": open(os.path.join(OUTPUT_DIR, "classification_hard_cot.jsonl"), "w", encoding="utf-8")
    }

    with open(temp_out, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                src = item.get("source", "")
                target = None
                if "pathmmu" in src: target = "pathmmu"
                elif "pathvqa" in src: target = "pathvqa"
                elif "classification" in src: target = "classification"
                
                if target:
                    files[target].write(line)
            except: continue

    for f in files.values():
        f.close()
    print("Done!")

def main():
    if not os.path.exists(HARD_SAMPLES_FILE):
        mining_phase()
    
    # Downsample
    # This recalculates every time? It's randomized. 
    # If `cot_temp_output` exists, we might duplicate if we re-sample differently?
    # We should cache downsampled list?
    # For robustness, we generate `downsampled_list.json` first?
    ds_file = os.path.join(OUTPUT_DIR, "downsampled_for_gen.json")
    if os.path.exists(ds_file):
        with open(ds_file, 'r') as f:
            inputs = json.load(f)
    else:
        inputs = downsample_and_split()
        with open(ds_file, 'w') as f:
            json.dump(inputs, f)
    generation_phase(inputs)
    split_results()

if __name__ == "__main__":
    main()
