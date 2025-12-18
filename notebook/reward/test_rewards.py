import sys
import os

# Add project root to sys.path to ensure swift can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swift.plugin.orm import VqaBleuReward, VqaBertReward

# Test cases
# Format: (prediction, reference, description)
test_cases = [
    # Similar pairs (Pathology domain)
    (
        "<answer>The image shows a histopathological section of renal cell carcinoma with clear cell type.</answer>",
        "<answer>This is a microscopic view of clear cell renal cell carcinoma.</answer>",
        "Similar: Renal Cell Carcinoma"
    ),
    (
        "<answer>Tumor cells are arranged in nests separated by delicate vascular network.</answer>",
        "<answer>Nests of tumor cells are separated by a fine network of blood vessels.</answer>",
        "Similar: Tumor Architecture"
    ),
    # Dissimilar pairs
    (
        "<answer>The tissue contains normal lung alveoli.</answer>",
        "<answer>The section reveals invasive ductal carcinoma of the breast.</answer>",
        "Dissimilar: Lung vs Breast Cancer"
    ),
    (
        "<answer>This is a case of acute inflammation with neutrophil infiltration.</answer>",
        "<answer>The slide shows chronic lymphocytic leukemia.</answer>",
        "Dissimilar: Inflammation vs Leukemia"
    )
]

def test_vqa_bleu():
    print("-" * 50)
    print("Testing VqaBleuReward (BLEU-4)")
    reward_fn = VqaBleuReward()
    
    predictions = [t[0] for t in test_cases]
    solutions = [t[1] for t in test_cases]
    tasks = ['vqa'] * len(test_cases)
    
    scores = reward_fn(predictions, solutions, task=tasks)
    
    for i, (score, case) in enumerate(zip(scores, test_cases)):
        print(f"Case {i+1} ({case[2]}):\n  Pred: {case[0]}\n  Ref:  {case[1]}\n  Score = {score:.4f}")

def test_vqa_bert():
    print("-" * 50)
    print("Testing VqaBertReward (BERT Similarity)")
    reward_fn = VqaBertReward()
    predictions = [t[0] for t in test_cases]
    solutions = [t[1] for t in test_cases]
    tasks = ['vqa'] * len(test_cases)
    scores = reward_fn(predictions, solutions, task=tasks)
    for i, (score, case) in enumerate(zip(scores, test_cases)):
        print(f"Case {i+1} ({case[2]}): Score = {score:.4f}")

if __name__ == "__main__":
    test_vqa_bleu()
    test_vqa_bert()
