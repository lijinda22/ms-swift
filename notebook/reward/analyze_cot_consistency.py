
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def load_data(file_path, limit=50):
    data = []
    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                item = json.loads(line)
                data.append(item)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return data

def get_prediction(model, tokenizer, device, premise, hypothesis, label_map):
    inputs = tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation='only_first').to(device)
    with torch.no_grad():
        logits = model(inputs)[0]
        probs = logits.softmax(dim=1).cpu().numpy()[0]
    predicted_index = np.argmax(probs)
    return label_map[predicted_index], probs

def plot_results(total, consistent, conflict, conflict_scores, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Pie Chart: Consistency Rate
    labels = ['Consistent', 'Conflict']
    sizes = [consistent, conflict]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  
    ax1.set_title(f'CoT Consistency Rate (N={total})')
    
    # 2. Histogram: Conflict Scores
    ax2.hist(conflict_scores, bins=20, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Min Contradiction Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Conflict Scores\n(Min Prob of Contradiction across 3 schemes)')
    ax2.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, 'cot_consistency_analysis.png')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def analyze_cot_consistency(model_path, data_file, output_dir, limit=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path}...")
    print(f"Device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    # 打印tokenizer 长度, 然后rasie
    # print(f"Tokenizer length: {len(tokenizer)}")
    # raise Exception("Tokenizer length: {len(tokenizer)}")
    model.eval()
    
    items = load_data(data_file, limit)
    if not items:
        return

    print("\n" + "="*160)
    print(f"{'Index':<6} | {'S1 Pred':<12} | {'S2 Pred':<12} | {'Final':<12} | {'S1 Probs':<20} | {'S2 Probs':<20}")
    print("-" * 160)

    label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    
    label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    
    inconsistent_items = []
    consistent_count = 0
    conflict_count = 0
    all_conflict_scores = []

    for idx, item in enumerate(items):
        question = item.get('question', '')
        cot = item.get('reasoning_cot_lingshu32b', '')
        correct_answer = item.get('correct_answer', '')
        
        # Scheme 1
        premise1 = f"{cot}"
        hypothesis1 = f"{question}\nThe answer is {correct_answer}."
        pred1, probs1 = get_prediction(model, tokenizer, device, premise1, hypothesis1, label_map)
        
        # Scheme 2
        premise2 = f"{question}\n{cot}"
        hypothesis2 = f"The answer is {correct_answer}."
        pred2, probs2 = get_prediction(model, tokenizer, device, premise2, hypothesis2, label_map)
        
        premise3 = f"{cot}"
        hypothesis3 = f"The answer is {correct_answer}."
        pred3, probs3 = get_prediction(model, tokenizer, device, premise3, hypothesis3, label_map)
        
        probs_str1 = f"{probs1[0]:.3f}/{probs1[1]:.3f}/{probs1[2]:.3f}"
        probs_str2 = f"{probs2[0]:.3f}/{probs2[1]:.3f}/{probs2[2]:.3f}"
        probs_str3 = f"{probs3[0]:.3f}/{probs3[1]:.3f}/{probs3[2]:.3f}"

        # 冲突分数: probs1, probs2 [2]
        conflict_score = min(probs1[2], probs2[2], probs3[2])
        
        # Logic: If either is NOT contradiction, then valid.
        if pred1 != 'contradiction' or pred2 != 'contradiction' or pred3 != 'contradiction':
            final_status = 'Consistent'
            consistent_count += 1
        else:
            final_status = 'CONFLICT'
            conflict_count += 1
            
        all_conflict_scores.append(conflict_score)
            
        print(f"{idx:<6} | {pred1:<12} | {pred2:<12} | {pred3:<12} | {final_status:<12} | {probs_str1:<20} | {probs_str2:<20} | {probs_str3:<20}")
        
        if final_status == 'CONFLICT':
            inconsistent_items.append({
                'index': idx,
                'pred1': pred1,
                'pred2': pred2,
                'pred3': pred3,
                'probs1': probs_str1,
                'probs2': probs_str2,
                'probs3': probs_str3,
                'question': question,
                'cot': cot,
                'answer': correct_answer
            })

    # Summary of Inconsistent Items
    if inconsistent_items:
        print("\n" + "="*80)
        print("SUMMARY OF INCONSISTENT (Both Schemes Contradict) ITEMS")
        print("="*80)
        for item in inconsistent_items:
            print(f"\n[Index: {item['index']}] S1: {item['pred1']} ({item['probs1']}) | S2: {item['pred2']} ({item['probs2']}) | S3: {item['pred3']} ({item['probs3']})")
            print(f"Question: {item['question'][:200]}..." if len(item['question'])>200 else f"Question: {item['question']}")
            print(f"Correct Answer: {item['answer']}")
            print("-" * 40)
            print(f"CoT (Reasoning):\n{item['cot']}")
            print("="*80)
    else:
        print("\n" + "="*80)
        print("No inconsistent items found (all passed at least one check).")
        print("="*80)
    print("inconsistent_items 数量: ", len(inconsistent_items), "/", len(items))
    
    plot_results(len(items), consistent_count, conflict_count, all_conflict_scores, output_dir)

if __name__ == "__main__":
    model_path = "/data/ckpt/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-mnli/"
    data_file = "/data/ljd/VLM-R1/dataset/sft/pathgen_instruct_close_cot_9144.jsonl"
    output_dir = "/data/ljd/Pathology_FM_LLM/expriment/reward/"
    
    print(f"Starting analysis on {data_file}")
    analyze_cot_consistency(model_path, data_file, output_dir, limit=10000)
