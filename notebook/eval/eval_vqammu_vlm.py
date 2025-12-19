import argparse
import json
import os
import re
import torch
import gc
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba
from vllm import LLM, SamplingParams

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

MODEL_PATHS = {
    "qwen3_vl-4b-instruct": "/data/ckpt/Qwen3-VL-4B-Instruct/",
    "qwen3_vl-4b-sft": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/sft_mmu/qwen3_vl_4b_sft/v0-20251217-094501/checkpoint-42-merged/",
    "qwen3_vl-4b-sft-kd-w0.2": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/sft_mmu/qwen3_vl_4b_sft_kd_0.2/v3-20251218-223605/checkpoint-42-merged/",
    "qwen3_vl-4b-sft-kd-w0.3": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/sft_mmu/qwen3_vl_4b_sft_kd_0.3/v0-20251219-093913/checkpoint-42-merged/",
    "qwen3_vl-4b-sft-kd-w0.5": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/sft_mmu/qwen3_vl_4b_sft_kd_0.5/v0-20251218-223652/checkpoint-42-merged/",
    "qwen3_vl-4b-sft-kd-w0.7": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/sft_mmu/qwen3_vl_4b_sft_kd_0.7/v0-20251219-100425/checkpoint-42-merged/",
}

PATHMMU_SOURCES = ["PubMed", "EduContent", "PathCLS", "Atlas"]

DATASET_CONFIGS = {
    # "pathvqa_closeset": {
    #     "path": "/data/dataset/vqa/path-vqa/data_refine/pathvqa_test_pathology.json",
    #     "type": "mcq",
    #     "filter_fn": lambda x: x.get("is_YORN", False) == True,
    # },
    # "pathvqa_openset": {
    #     "path": "/data/dataset/vqa/path-vqa/data_refine/pathvqa_test_pathology.json",
    #     "type": "vqa",
    #     "filter_fn": lambda x: x.get("is_YORN", False) == False,
    # },
}

for source in PATHMMU_SOURCES:
    for split in ["val", "test_tiny"]:
        dataset_name = f"pathmmu_{source}_{split}"
        DATASET_CONFIGS[dataset_name] = {
            "path": "/data/ljd/VLM-R1/dataset/sft/pathmmu.json",
            "type": "mcq",
            "split": split,
            "source": source,
        }

OUTPUT_BASE_DIR = "/data/ljd/Pathology_FM_LLM/expriment/eval_results/mmu_test"

def load_pathvqa_dataset(config):
    with open(config["path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    if "filter_fn" in config:
        data = [item for item in data if config["filter_fn"](item)]
    return data

def load_pathmmu_dataset(config):
    with open(config["path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    split = config.get("split", "none")
    target_source = config.get("source", None)
    all_items = []
    for source_name, source_data in data.items():
        if target_source and source_name != target_source:
            continue
        if split in source_data:
            for item in source_data[split]:
                item["source"] = source_name
                all_items.append(item)
    return all_items

def load_dataset(dataset_name):
    if dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        if "pathmmu" in dataset_name:
            data = load_pathmmu_dataset(config)
        else:
            data = load_pathvqa_dataset(config)
        return data, config["type"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size if hasattr(processor, "image_processor") else None,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    return {'prompt': text, 'multi_modal_data': mm_data, 'mm_processor_kwargs': video_kwargs}

def init_vllm_model(model_key):
    model_path = MODEL_PATHS[model_key]
    print(f"Loading {model_path}...")
    llm = LLM(model=model_path, mm_encoder_tp_mode="data", tensor_parallel_size=torch.cuda.device_count(), seed=0, gpu_memory_utilization=0.9, max_model_len=8192)
    processor = AutoProcessor.from_pretrained(model_path)
    return llm, processor

def parse_mcq_answer(output_text):
    parsed = output_text.strip()
    if "<answer>" in parsed:
        parsed = parsed.split("<answer>")[-1].split("</answer>")[0].strip()
    match = re.search(r'\b([A-F])\b', parsed, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if parsed and parsed[0].isalpha():
        return parsed[0].upper()
    return parsed

def calculate_bleu4(reference, hypothesis):
    # Tokenize using jieba, consistent with swift/plugin/orm.py
    hyp_tokens = list(jieba.cut(hypothesis))
    ref_tokens = list(jieba.cut(reference))
    
    if not hyp_tokens or not ref_tokens:
        return 0.0
        
    smoothing = SmoothingFunction().method3
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)

def eval_dataset_vllm(llm, processor, data, dataset_name, dataset_type, model_key, batch_size=16):
    
    output_dir = os.path.join(OUTPUT_BASE_DIR, model_key)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(os.path.join(output_dir, f"{dataset_name}_results.json")):
        return
    sampling_params = SamplingParams(temperature=0, max_tokens=512, top_k=-1, stop_token_ids=[])
    correct = 0
    exact_match_count = 0
    bleu_scores = []
    total = 0
    results = []
    
    for i in tqdm(range(0, len(data), batch_size), desc=f"Evaluating {dataset_name}"):
        batch_items = data[i : i + batch_size]
        batch_inputs = []
        batch_metadata = []
        for item in batch_items:
            image_path = item.get("image", item.get("img", item.get("image_path")))
            question = item["question"]
            gt_answer = item["answer"]
            
            if dataset_type == "mcq":
                 question_text = f"{question.strip()}\nPlease output only the final answer option directly. Just one letter (A, B, C, or D) with no explanation or additional text."
            else:
                 question_text = question
            
            messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question_text}]}]
            try:
                batch_inputs.append(prepare_inputs_for_vllm(messages, processor))
                batch_metadata.append({"image_path": image_path, "question": question, "gt_answer": gt_answer})
            except Exception:
                continue
        
        if not batch_inputs: continue
        try:
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
        except Exception:
            continue

        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            metadata = batch_metadata[j]
            gt_answer = metadata["gt_answer"]
            
            if dataset_type == "mcq":
                parsed_answer = parse_mcq_answer(generated_text)
                is_correct = (parsed_answer == gt_answer.upper())
                if is_correct: correct += 1
                results.append({**metadata, "prediction": generated_text, "parsed_answer": parsed_answer, "is_correct": is_correct})
            elif dataset_type == "vqa":
                is_exact_match = (generated_text.strip().lower() == gt_answer.strip().lower())
                if is_exact_match: exact_match_count += 1
                bleu_score = calculate_bleu4(gt_answer, generated_text)
                bleu_scores.append(bleu_score)
                results.append({**metadata, "prediction": generated_text, "exact_match": is_exact_match, "bleu4": bleu_score})
            total += 1
            
    metrics = {}
    if dataset_type == "mcq":
        metrics["accuracy"] = correct / total if total > 0 else 0
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    elif dataset_type == "vqa":
        metrics["exact_match"] = exact_match_count / total if total > 0 else 0
        metrics["bleu4"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        print(f"Exact Match: {metrics['exact_match']:.4f}, BLEU-4: {metrics['bleu4']:.4f}")
        
    with open(os.path.join(output_dir, f"{dataset_name}_results.json"), "w", encoding="utf-8") as f:
        json.dump({"model": model_key, "dataset": dataset_name, "metrics": metrics, "results": results}, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    models = [args.model] if args.model else list(MODEL_PATHS.keys())
    datasets = [args.dataset] if args.dataset else list(DATASET_CONFIGS.keys())
    
    for model_key in models:
        try:
            llm, processor = init_vllm_model(model_key)
            for dataset_name in datasets:
                output_file = os.path.join(OUTPUT_BASE_DIR, model_key, f"{dataset_name}_results.json")
                if os.path.exists(output_file):
                    print(f"Skipping {dataset_name} for {model_key}, exists.")
                    continue
                print(f"Evaluating {dataset_name}...")
                data, dataset_type = load_dataset(dataset_name)
                if data: eval_dataset_vllm(llm, processor, data, dataset_name, dataset_type, model_key, args.batch_size)
            del llm, processor
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error {model_key}: {e}")
            
    # Aggregate results to CSV
    import csv
    print("Aggregating results to CSV...")
    csv_file = os.path.join(OUTPUT_BASE_DIR, "vqammu_summary.csv")
    
    # Collect all results
    summary_data = []
    datasets = list(DATASET_CONFIGS.keys())
    
    for model_key in models:
        row = {"Model": model_key}
        for dataset_name in datasets:
            result_file = os.path.join(OUTPUT_BASE_DIR, model_key, f"{dataset_name}_results.json")
            if os.path.exists(result_file):
                with open(result_file, "r", encoding="utf-8") as f:
                    try:
                        res = json.load(f)
                        metrics = res.get("metrics", {})
                        if "accuracy" in metrics:
                            row[dataset_name] = f"{metrics['accuracy']:.4f}"
                        elif "exact_match" in metrics:
                            row[dataset_name] = f"EM:{metrics['exact_match']:.4f}/B4:{metrics['bleu4']:.4f}"
                        else:
                            row[dataset_name] = "N/A"
                    except:
                        row[dataset_name] = "Error"
            else:
                row[dataset_name] = "-"
        summary_data.append(row)
        
    if summary_data:
        # Write to CSV
        fieldnames = ["Model"] + datasets
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)
        print(f"Summary saved to {csv_file}")

if __name__ == "__main__":
    main()
