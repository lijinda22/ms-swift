# -*- coding: utf-8 -*-
import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

# Set environment variable for VLLM
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# Model paths configuration
MODEL_PATHS = {
    # "qwen2.5_vl-7b": "/data/ckpt/Qwen2.5-VL-7B-Instruct/",
    # "patho-r1-7b": "/data/ckpt/Patho-R1-7B",
    # "lingshu-7b": "/data/ckpt/Lingshu-7B/",
    # "lingshu-32b": "/data/ckpt/Lingshu-32B/",
    "qwen3_vl-2b": "/data/ckpt/Qwen3-VL-2B-Instruct/",
}

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ required
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
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

def eval_model(model_key, test_file):
    model_path = MODEL_PATHS[model_key]
    print(f"Loading model from {model_path}...")
    
    # Initialize VLLM
    llm = LLM(
        model=model_path,
        mm_encoder_tp_mode="data",
        # enable_expert_parallel=True, # Qwen2.5/3-VL are usually dense models
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
        gpu_memory_utilization=0.9, # Adjust if needed
        max_model_len=8192, # Adjust based on model config if needed
    )
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=512, # Matches the max_new_tokens in original script
        top_k=-1,
        stop_token_ids=[],
    )
    
    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r') as f:
        data = json.load(f)
        
    correct = 0
    total = 0
    results = []
    
    print("Starting evaluation...")
    
    # Process all data to prepare inputs
    # VLLM can handle batching, so we can pass a list of inputs.
    # However, for very large datasets, we might want to chunk it to avoid preparing all inputs in memory at once if images are loaded?
    # process_vision_info usually loads images if they are local paths.
    # Let's do it in chunks to be safe and consistent with the original script's batching, 
    # but we can pass larger batches to vllm.generate.
    
    batch_size = 16 # VLLM can handle larger batches usually
    
    for i in tqdm(range(0, len(data), batch_size)):
        batch_items = data[i : i + batch_size]
        batch_inputs = []
        batch_metadata = [] # Store metadata to map results back
        
        for item in batch_items:
            image_path = item['image_path']
            question = item['question']
            gt_answer = item['answer']
            
            if model_key == "patho-r1-7b":
                question_text = question
                # question_text = f"{question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format. Just one letter (A, B, C, or D) with no explanation or additional text in <answer> </answer>."
            else:
                question_text = f"{question} Please output only the final answer option directly. Just one letter (A, B, C, or D) with no explanation or additional text."
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {
                            "type": "text",
                            "text": question_text,
                        },
                    ],
                }
            ]
            
            try:
                vllm_input = prepare_inputs_for_vllm(messages, processor)
                batch_inputs.append(vllm_input)
                batch_metadata.append({
                    "image_path": image_path,
                    "question": question_text,
                    "gt_answer": gt_answer
                })
            except Exception as e:
                print(f"Error preparing input for {image_path}: {e}")
                continue
        
        if not batch_inputs:
            continue
            
        # Inference
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
        
        # Process results
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            metadata = batch_metadata[j]
            
            gt_answer = metadata['gt_answer']
            image_path = metadata['image_path']
            question = metadata['question']
            
            # Basic parsing logic
            parsed_answer = generated_text.strip()
            if "<answer>" in parsed_answer:
                parsed_answer = parsed_answer.split("<answer>")[-1].split("</answer>")[0].strip()
            
            is_correct = False
            if gt_answer.lower() in parsed_answer.lower():
                 # Refine parsing
                 import re
                 match = re.search(r'\b([A-D])\b', parsed_answer)
                 if match:
                     predicted_letter = match.group(1)
                     if predicted_letter == gt_answer:
                         is_correct = True
                 elif parsed_answer.strip() == gt_answer:
                     is_correct = True
            
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "image_path": image_path,
                "question": question,
                "gt_answer": gt_answer,
                "prediction": generated_text,
                "parsed_answer": parsed_answer,
                "is_correct": is_correct
            })
        
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    output_file = f"/data/ljd/Pathology_FM_LLM/expriment/classify/CCRCC/vlm/test_results_{model_key}_vllm.json"
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLM on CCRCC dataset with VLLM")
    parser.add_argument("--model", type=str, default=None, choices=MODEL_PATHS.keys(), help="Model key to evaluate. If not specified, runs all models.")
    parser.add_argument("--test_file", type=str, default="/data/ljd/Pathology_FM_LLM/expriment/classify/CCRCC/test.json", help="Path to test.json")
    
    args = parser.parse_args()
    
    if args.model:
        models_to_run = [args.model]
    else:
        models_to_run = list(MODEL_PATHS.keys())
    
    import gc
    
    for model_key in models_to_run:
        print(f"\n\n{'='*20} Evaluating {model_key} {'='*20}")
        # Note: VLLM might not clean up GPU memory fully between runs in the same process easily without destroying the LLM object.
        # But we are creating a new LLM object in eval_model.
        # Ideally, we should run each model in a separate process if we want to be safe, 
        # but let's try this.
        try:
            eval_model(model_key, args.test_file)
        except Exception as e:
            print(f"Error evaluating {model_key}: {e}")
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
