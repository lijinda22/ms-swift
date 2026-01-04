import argparse
import json
import os
import re
import torch
import gc
import csv
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba
import random
from vllm import LLM, SamplingParams
import sys

# Ensure we can import swift modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from swift.plugin.orm import VqaBertReward

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATHS = {
    "Patho-R1-7B": "/data/ckpt/Patho-R1-7B/",
    # "Patho-adapter-grpo": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/grpo/qwen3_vl_4b_cpt_sft_kd_mmu_lr5e-6_old/v0-20251230-144744/checkpoint-1200-merged/",
}
COT_QUESTION_TEMPLATE = "{Question}\nThink through the question step by step, enclose your reasoning process in <think>...</think> tags. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags. No extra information or text outside of these tags."
OPEN_COT_QUESTION_TEMPLATE = "{Question}\nThink through the question step by step, enclose your reasoning process in <think>...</think> tags. Then provide the answer inside <answer>...</answer> tags. No extra information or text outside of these tags."

OUTPUT_BASE_DIR = "/data/ljd/Pathology_FM_LLM/expriment/eval_results/rl_test"

# ============================================================================
# Dataset Loading
# ============================================================================

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

def load_pathvqa_dataset(config: Dict) -> List[Dict]:
    with open(config["path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    if "filter_fn" in config:
        data = [item for item in data if config["filter_fn"](item)]
    return data

def load_dataset_by_name(dataset_name: str) -> Tuple[List[Dict], str]:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    config = DATASET_CONFIGS[dataset_name]
    return load_pathvqa_dataset(config), config["type"]

# ============================================================================
# Evaluation Engine
# ============================================================================

class Evaluator:
    def __init__(self, model_key: str, gpu_util: float = 0.9):
        self.model_key = model_key
        self.model_path = MODEL_PATHS[model_key]
        print(f"Initializing model: {model_key} from {self.model_path}")
        
        self.llm = LLM(
            model=self.model_path,
            mm_encoder_tp_mode="data",
            tensor_parallel_size=torch.cuda.device_count(),
            seed=0,
            gpu_memory_utilization=gpu_util,
            max_model_len=4096
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.bert_reward = None # Lazy load

    def _get_bert_reward(self):
        if self.bert_reward is None:
            self.bert_reward = VqaBertReward()
        return self.bert_reward

    def prepare_inputs(self, messages):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size if hasattr(self.processor, "image_processor") else None,
            return_video_kwargs=True,
            return_video_metadata=True
        )
        mm_data = {}
        if image_inputs is not None: mm_data['image'] = image_inputs
        if video_inputs is not None: mm_data['video'] = video_inputs
        return {'prompt': text, 'multi_modal_data': mm_data, 'mm_processor_kwargs': video_kwargs}

    @staticmethod
    def parse_mcq_answer(output_text: str) -> str:
        parsed = output_text.strip()
        if "<answer>" in parsed:
            parsed = parsed.split("<answer>")[-1].split("</answer>")[0].strip()
        match = re.search(r'\b([A-Z])\b', parsed, re.IGNORECASE)
        if match: return match.group(1).upper()
        if parsed and parsed[0].isalpha(): return parsed[0].upper()
        return parsed

    @staticmethod
    def extract_answer_content(text: str) -> str:
        if "<answer>" in text and "</answer>" in text:
            return text.split("<answer>")[-1].split("</answer>")[0].strip()
        return text.strip()

    @staticmethod
    def calculate_bleu4(reference: str, hypothesis: str) -> float:
        hyp_tokens = list(jieba.cut(hypothesis))
        ref_tokens = list(jieba.cut(reference))
        if not hyp_tokens or not ref_tokens: return 0.0
        smoothing = SmoothingFunction().method3
        return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)

    def evaluate_dataset(self, dataset_name: str, data: List[Dict], dataset_type: str, batch_size: int = 16):
        output_dir = os.path.join(OUTPUT_BASE_DIR, self.model_key)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{dataset_name}_results.json")
        
        if os.path.exists(output_file):
            print(f"Skipping {dataset_name} for {self.model_key}, already exists.")
            return

        sampling_params = SamplingParams(temperature=0, max_tokens=512, top_k=-1)
        correct, exact_match, total = 0, 0, 0
        bleu_scores, bert_scores = [], []
        results = []

        print(f"Running inference for {dataset_name}...")
        for i in tqdm(range(0, len(data), batch_size), desc=f"Evaluating {dataset_name}"):
            batch_items = data[i : i + batch_size]
            batch_inputs, batch_metadata = [], []
            
            for item in batch_items:
                image_path = item.get("image", item.get("img", item.get("image_path")))
                question = item["question"]
                gt_answer = item["answer"]
                
                if dataset_type == "vqa":
                    question_text = OPEN_COT_QUESTION_TEMPLATE.format(Question=question.strip())
                else: 
                    question_text = COT_QUESTION_TEMPLATE.format(Question=question.strip())
                messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question_text}]}]
                
                try:
                    batch_inputs.append(self.prepare_inputs(messages))
                    batch_metadata.append({"image_path": image_path, "question": question, "gt_answer": gt_answer})
                except Exception: continue
            
            if not batch_inputs: continue
            try:
                outputs = self.llm.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
            except Exception: continue

            for j, output in enumerate(outputs):
                gen_text = output.outputs[0].text
                meta = batch_metadata[j]
                gt = meta["gt_answer"]
                
                res_item = {**meta, "prediction": gen_text}
                
                if dataset_type == "mcq":
                    parsed = self.parse_mcq_answer(gen_text)
                    is_correct = (parsed == gt.upper())
                    if is_correct: correct += 1
                    res_item.update({"parsed_answer": parsed, "is_acc": is_correct})
                else:
                    # EM / BLEU-4 / BERT Score for Open-set VQA
                    answer_content = self.extract_answer_content(gen_text)
                    is_em = (answer_content.lower() == gt.strip().lower())
                    if is_em: exact_match += 1
                    bleu = self.calculate_bleu4(gt, answer_content)
                    bleu_scores.append(bleu)
                    
                    bert_s = 0.0
                    try:
                        b_scores = self._get_bert_reward()([gen_text], [f"<answer>{gt}</answer>"], task=['vqa'])
                        bert_s = b_scores[0] if b_scores and b_scores[0] is not None else 0.0
                    except: pass
                    bert_scores.append(bert_s)
                    res_item.update({"exact_match": is_em, "bleu4": bleu, "bert_score": bert_s})
                
                total += 1
                results.append(res_item)

        metrics = {}
        if dataset_type == "mcq":
            metrics["accuracy"] = correct / total if total > 0 else 0
            print(f"[{dataset_name}] Accuracy: {metrics['accuracy']:.4f}")
        else:
            metrics["exact_match"] = exact_match / total if total > 0 else 0
            metrics["bleu4"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
            metrics["bert_score"] = sum(bert_scores) / len(bert_scores) if bert_scores else 0
            print(f"[{dataset_name}] EM: {metrics['exact_match']:.4f}, BLEU4: {metrics['bleu4']:.4f}, BERT: {metrics['bert_score']:.4f}")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "model": self.model_key, 
                "dataset": dataset_name, 
                "metrics": metrics, 
                "results": results 
            }, f, indent=2, ensure_ascii=False)
        return metrics

def main():
    print("Available models:", list(MODEL_PATHS.keys()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--gpu_util", type=float, default=0.65)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()
    
    selected_models = [args.model] if args.model else list(MODEL_PATHS.keys())
    selected_datasets = [args.dataset] if args.dataset else list(DATASET_CONFIGS.keys())
    
    for model_key in selected_models:
        datasets_to_run = [ds for ds in selected_datasets if not os.path.exists(os.path.join(OUTPUT_BASE_DIR, model_key, f"{ds}_results.json"))]
        
        if not datasets_to_run:
            print(f"All datasets for {model_key} already evaluated.")
            continue

        evaluator = Evaluator(model_key, args.gpu_util)
        for ds_name in datasets_to_run:
            print(f"\n>>>> Evaluating {ds_name}")
            data, ds_type = load_dataset_by_name(ds_name)
            if args.limit:
                data = random.sample(data, min(len(data), args.limit))
            
            if data:
                evaluator.evaluate_dataset(ds_name, data, ds_type, args.batch_size)
        
        del evaluator.llm
        del evaluator.processor
        if evaluator.bert_reward: del evaluator.bert_reward
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
