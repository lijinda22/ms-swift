#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive evaluation script for pathology VLM models using vLLM for inference.
Supports multiple datasets: PathVQA (close-set/open-set), PathMMU, and classification datasets.
"""

import argparse
import json
import os
import re
import torch
import gc
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random

# vLLM imports
from vllm import LLM, SamplingParams

# Set environment variable for VLLM
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# ============================================================================
# Prompt Templates (matching training data format)
# ============================================================================

CLOSE_QUESTION_TEMPLATE = "{Question}\nPlease output only the final answer option directly. Just one letter (A, B, C, or D) with no explanation or additional text."

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATHS = {
    "qwen3_vl-2b-instruct": "/data/ckpt/Qwen3-VL-2B-Instruct/",
    # "qwen3_vl-2b-think": "/data/ckpt/Qwen3-VL-2B-Thinking/",
    "qwen3_vl-2b-sft": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/qwen3_vl_2b_sft/v5-20251213-130919/checkpoint-1242-merged/",
    "qwen3_vl-2b-sft-kd-w0.5": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/qwen3_vl_2b_sft_kd_0.5/v0-20251213-135528/checkpoint-1242-merged/",
    "qwen3_vl-2b-cpt_sft": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/qwen3_vl_2b_cpt_sft_20k/v0-20251214-121237/checkpoint-1242-merged/",
    "qwen3_vl-2b-cpt_sft-kd-w0.5": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/qwen3_vl_2b_cpt_sft_20k_kd_0.5/v0-20251214-121311/checkpoint-1242-merged/",
    # Add other models as needed
    # "patho-r1-7b": "/data/ckpt/Patho-R1-7B",
}

# PathMMU sources will be dynamically loaded
PATHMMU_SOURCES = ["PubMed", "EduContent", "PathCLS", "Atlas"]

DATASET_CONFIGS = {
    "pathvqa_closeset": {
        "path": "/data/dataset/vqa/path-vqa/data_refine/pathvqa_test_pathology.json",
        "type": "mcq",
        "filter_fn": lambda x: x.get("is_YORN", False) == True,
    },
    "pathvqa_openset": {
        "path": "/data/dataset/vqa/path-vqa/data_refine/pathvqa_test_pathology.json",
        "type": "vqa",
        "filter_fn": lambda x: x.get("is_YORN", False) == False,
    },
}

# Dynamically add PathMMU datasets for each source and split
for source in PATHMMU_SOURCES:
    for split in ["test", "test_tiny"]:
        dataset_name = f"pathmmu_{source}_{split}"
        DATASET_CONFIGS[dataset_name] = {
            "path": "/data/ljd/VLM-R1/dataset/sft/pathmmu.json",
            "type": "mcq",
            "split": split,
            "source": source,
        }

CLASSIFICATION_DATASETS = [
    "CCRCC", "BreaKHis", "chaoyang", "crc100k", "CRC_MSI", "PanCancer-TIL"
]
CLASSIFICATION_BASE_DIR = "/data/ljd/Pathology_FM_LLM/expriment/classify"

OUTPUT_BASE_DIR = "/data/ljd/Pathology_FM_LLM/expriment/eval_results"

# ============================================================================
# Dataset Loaders
# ============================================================================

def load_pathvqa_dataset(config: Dict) -> List[Dict]:
    """Load PathVQA dataset with optional filtering."""
    with open(config["path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "filter_fn" in config:
        data = [item for item in data if config["filter_fn"](item)]
    
    return data


def load_pathmmu_dataset(config: Dict) -> List[Dict]:
    """Load PathMMU dataset for a specific split and optionally a specific source."""
    with open(config["path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    
    split = config.get("split", "test")
    target_source = config.get("source", None)  # If None, load all sources
    all_items = []
    
    # PathMMU has nested structure: {source_name: {split: [items]}}
    for source_name, source_data in data.items():
        # Skip if we're filtering by source and this isn't the target
        if target_source and source_name != target_source:
            continue
            
        if split in source_data:
            for item in source_data[split]:
                item["source"] = source_name
                all_items.append(item)
    
    return all_items


def load_classification_dataset(dataset_name: str) -> List[Dict]:
    """Load a single classification dataset."""
    test_file = os.path.join(CLASSIFICATION_BASE_DIR, dataset_name, "test.json")
    
    if not os.path.exists(test_file):
        print(f"Warning: {test_file} not found. Skipping.")
        return []
    
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def load_dataset(dataset_name: str) -> Tuple[List[Dict], str]:
    """
    Load dataset by name.
    Returns: (data, dataset_type)
    """
    if dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        if "pathmmu" in dataset_name:
            data = load_pathmmu_dataset(config)
        else:
            data = load_pathvqa_dataset(config)
        return data, config["type"]
    
    elif dataset_name in CLASSIFICATION_DATASETS:
        data = load_classification_dataset(dataset_name)
        return data, "mcq"
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================================================
# vLLM Handling
# ============================================================================

def prepare_inputs_for_vllm(messages, processor):
    """Prepare inputs for vLLM generation."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size if hasattr(processor, "image_processor") else None, # Some processors might differ
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

def init_vllm_model(model_key: str):
    """Initialize vLLM model and processor."""
    model_path = MODEL_PATHS[model_key]
    print(f"Loading model from {model_path} with vLLM...")
    
    llm = LLM(
        model=model_path,
        mm_encoder_tp_mode="data",
        # enable_expert_parallel=False, # Default is False, explicit for clarity if needed, but removing to avoid pydantic errors on dense models
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
    )
    
    processor = AutoProcessor.from_pretrained(model_path)
    return llm, processor


# ============================================================================
# Answer Parsing
# ============================================================================

def parse_mcq_answer(output_text: str) -> str:
    """Parse multiple-choice answer from model output."""
    parsed = output_text.strip()
    
    # Handle <answer> tags
    if "<answer>" in parsed:
        parsed = parsed.split("<answer>")[-1].split("</answer>")[0].strip()
    
    # Extract single letter (A/B/C/D/E/F...)
    match = re.search(r'\b([A-F])\b', parsed, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback: return first character if it's a letter
    if parsed and parsed[0].isalpha():
        return parsed[0].upper()
    
    return parsed


def calculate_bleu4(reference: str, hypothesis: str) -> float:
    """Calculate BLEU-4 score."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    smoothing = SmoothingFunction().method1
    score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
    return score


# ============================================================================
# Evaluation Functions
# ============================================================================

def eval_dataset_vllm(
    llm: LLM,
    processor: Any,
    data: List[Dict],
    dataset_name: str,
    dataset_type: str,
    model_key: str,
    batch_size: int = 16
):
    """Evaluate dataset using vLLM."""
    
    # Prepare Sampling Params
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=512,
        top_k=-1,
        stop_token_ids=[],
    )

    correct = 0
    exact_match_count = 0
    bleu_scores = []
    total = 0
    results = []
    
    print(f"evaluating {len(data)} samples...")
    
    # Iterate in batches
    for i in tqdm(range(0, len(data), batch_size), desc=f"Evaluating {dataset_name}"):
        batch_items = data[i : i + batch_size]
        batch_inputs = []
        batch_metadata = []
        
        for item in batch_items:
            # Handle different image path keys
            image_path = item.get("image", item.get("img", item.get("image_path")))
            question = item["question"]
            gt_answer = item["answer"]
            
            # Format question
            if dataset_type == "mcq":
                 question_text = CLOSE_QUESTION_TEMPLATE.format(Question=question.strip())
            else:
                 question_text = question # Open set VQA usually uses the question directly
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question_text},
                    ],
                }
            ]
            
            try:
                vllm_input = prepare_inputs_for_vllm(messages, processor)
                batch_inputs.append(vllm_input)
                batch_metadata.append({
                    "image_path": image_path,
                    "question": question,
                    "gt_answer": gt_answer,
                })
            except Exception as e:
                print(f"Error preparing input for {image_path}: {e}")
                continue
        
        if not batch_inputs:
            continue
            
        # Inference
        try:
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
        except Exception as e:
            print(f"Error during generation: {e}")
            continue

        # Process results
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            metadata = batch_metadata[j]
            gt_answer = metadata["gt_answer"]
            
            # Metrics calculation
            if dataset_type == "mcq":
                parsed_answer = parse_mcq_answer(generated_text)
                is_correct = (parsed_answer == gt_answer.upper())
                if is_correct:
                    correct += 1
                
                results.append({
                    "image_path": metadata["image_path"],
                    "question": metadata["question"],
                    "gt_answer": gt_answer,
                    "prediction": generated_text,
                    "parsed_answer": parsed_answer,
                    "is_correct": is_correct
                })
                
            elif dataset_type == "vqa":
                # Exact match
                is_exact_match = (generated_text.strip().lower() == gt_answer.strip().lower())
                if is_exact_match:
                    exact_match_count += 1
                    
                # BLEU-4
                bleu_score = calculate_bleu4(gt_answer, generated_text)
                bleu_scores.append(bleu_score)
                
                results.append({
                    "image_path": metadata["image_path"],
                    "question": metadata["question"],
                    "gt_answer": gt_answer,
                    "prediction": generated_text,
                    "exact_match": is_exact_match,
                    "bleu4": bleu_score
                })
            
            total += 1
            
    # Final Metrics
    metrics = {}
    if dataset_type == "mcq":
        accuracy = correct / total if total > 0 else 0
        metrics["accuracy"] = accuracy
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    elif dataset_type == "vqa":
        em_score = exact_match_count / total if total > 0 else 0
        bleu_avg = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        metrics["exact_match"] = em_score
        metrics["bleu4"] = bleu_avg
        print(f"Exact Match: {em_score:.4f}")
        print(f"BLEU-4: {bleu_avg:.4f}")
        
    # Save results
    output_dir = os.path.join(OUTPUT_BASE_DIR, model_key)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_results_vllm.json") # Updated filename to indicate vLLM
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_key,
            "dataset": dataset_name,
            "dataset_type": dataset_type,
            "num_samples": total,
            "metrics": metrics,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM models on pathology datasets using vLLM")
    parser.add_argument("--model", type=str, default=None, choices=list(MODEL_PATHS.keys()),
                        help="Model to evaluate.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset to evaluate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    # Determine models
    if args.model:
        models_to_eval = [args.model]
    else:
        models_to_eval = list(MODEL_PATHS.keys())
    
    # Determine datasets
    if args.dataset:
        datasets_to_eval = [args.dataset]
    else:
        datasets_to_eval = list(DATASET_CONFIGS.keys()) + CLASSIFICATION_DATASETS
    
    # Run evaluation
    for model_key in models_to_eval:
        print(f"\n\n{'='*60}")
        print(f"Initializing Model: {model_key}")
        print(f"{'='*60}\n")
        
        try:
            llm, processor = init_vllm_model(model_key)
            
            for dataset_name in datasets_to_eval:
                # Check if results already exist
                output_dir = os.path.join(OUTPUT_BASE_DIR, model_key)
                output_file = os.path.join(output_dir, f"{dataset_name}_results_vllm.json")
                
                if os.path.exists(output_file):
                    print(f"Results for {model_key} on {dataset_name} already exist at {output_file}. Skipping.")
                    continue

                print(f"\nEvaluating on {dataset_name}...")
                data, dataset_type = load_dataset(dataset_name)
                
                if not data:
                    print(f"No data for {dataset_name}, skipping.")
                    continue

                if args.debug:
                    print("Debug mode: limiting to 20 samples")
                    data = random.sample(data, min(len(data), 20))
                
                eval_dataset_vllm(llm, processor, data, dataset_name, dataset_type, model_key, args.batch_size)
                
            # Cleanup efforts
            del llm
            del processor
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to evaluate model {model_key}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
