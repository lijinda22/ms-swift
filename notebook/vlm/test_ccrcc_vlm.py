import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# Model paths configuration
MODEL_PATHS = {
    # "qwen2.5_vl-7b": "/data/ckpt/Qwen2.5-VL-7B-Instruct/",
    "patho-r1-7b": "/data/ckpt/Patho-R1-7B",
    # "lingshu-7b": "/data/ckpt/Lingshu-7B/",
    # "lingshu-32b": "/data/ckpt/Lingshu-32B/",
    # "qwen3_vl-2b": "/data/ckpt/Qwen3-VL-2B-Instruct/",
}

def load_model_and_processor(model_key):
    model_path = MODEL_PATHS[model_key]
    print(f"Loading model from {model_path}...")
    
    # if "qwen3" in model_key:
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    # else:
    #     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #         model_path,
    #         dtype=torch.bfloat16,
    #         attn_implementation="flash_attention_2",
    #         device_map="auto",
    #     )
    
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def eval_model(model_key, test_file):
    model, processor = load_model_and_processor(model_key)
    
    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r') as f:
        data = json.load(f)
        
    correct = 0
    total = 0
    results = []
    
    print("Starting evaluation...")
    print("Starting evaluation...")
    batch_size = 4
    for i in tqdm(range(0, len(data), batch_size)):
        batch_items = data[i : i + batch_size]
        batch_messages = []
        batch_questions = []
        batch_gt_answers = []
        batch_image_paths = []
        
        for item in batch_items:
            image_path = item['image_path']
            question = item['question']
            gt_answer = item['answer']
            
            batch_image_paths.append(image_path)
            batch_gt_answers.append(gt_answer)
            
            if model_key == "patho-r1-7b":
                question_text = question
                # question_text = f"{question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format. Just one letter (A, B, C, or D) with no explanation or additional text in <answer> </answer>."
            else:
                question_text = f"{question} Please output only the final answer option directly. Just one letter (A, B, C, or D) with no explanation or additional text."
            
            batch_questions.append(question_text)
            
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
            batch_messages.append(messages)
        
        inputs = processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        )
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=512, use_cache=True, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
            
        # Process results
        for j, output_text in enumerate(output_texts):
            gt_answer = batch_gt_answers[j]
            image_path = batch_image_paths[j]
            question = batch_questions[j] # Note: this is the modified question text
            
            # Basic parsing logic
            parsed_answer = output_text.strip()
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
                "prediction": output_text,
                "parsed_answer": parsed_answer,
                "is_correct": is_correct
            })
        
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    output_file = f"/data/ljd/Pathology_FM_LLM/expriment/classify/CCRCC/vlm/test_results_{model_key}.json"
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLM on CCRCC dataset")
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
        eval_model(model_key, args.test_file)
        
        # Cleanup to avoid OOM
        gc.collect()
        torch.cuda.empty_cache()
