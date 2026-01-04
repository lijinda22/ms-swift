import sys
import os
import json
import re

# Add project root to sys.path to ensure swift can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swift.plugin.orm import AccuracyBleuReward, AccuracyBertReward, CoTConsistencyReward, ConchGLiNERReward

def test_complex_rewards():
    print("=" * 60)
    print("Testing Complex Rewards (Accuracy, CoT Consistency, ConchGLiNER)")
    print("=" * 60)

    # Sample Data provided by user
    data = {
        "messages": [
            {
                "role": "user",
                "content": "<image>\nBased on the observed morphology, what type of cells are most likely represented by the small round to oval structures?\nA. Adipocytes\nB. Nerve cells\nC. Fibroblasts\nD. Lymphocytes\nThink through the question step by step, enclose your reasoning process in <think>...</think> tags. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags. No extra information or text outside of these tags."
            },
            {
                "role": "assistant",
                "content": "<think>Upon examining the medical image, the small round to oval structures are dispersed throughout the tissue. These structures are relatively uniform in size and shape, which is characteristic of fibroblasts. Fibroblasts are known for their role in producing extracellular matrix and collagen, which can be inferred from the dense, fibrous appearance of the surrounding tissue. Adipocytes typically have a larger, more irregular shape due to lipid content, which is not observed here. Nerve cells would exhibit distinct morphological features such as axons and dendrites, which are absent in this image. Lymphocytes are generally smaller and more densely packed, often seen in clusters, which does not match the observed pattern. Therefore, the morphology and distribution of these cells strongly suggest they are fibroblasts.\n### Final Answer: C. Fibroblasts</think> <answer>C</answer>"
            }
        ],
        "images": ["/nvme/pathgen/patch_pairs/TCGA-RA-A741/TCGA-RA-A741_21696_46720.png"],
        "source": "pathgen_instruct_close_cot"
    }

    # Extract components
    user_msg = data['messages'][0]['content']
    assistant_msg = data['messages'][1]['content']
    
    # 1. Prediction and Solution
    # In a real training scenario, prediction is generated. Here we treat the assistant message as the prediction AND the solution 
    # (or we can pretend we have a ground truth).
    # Let's assume the assistant message is the "Prediction" we want to score against itself as "Ground Truth" (expected 1.0)
    # OR better, let's treat it as a perfect prediction.
    # pred = GT
    # prediction = assistant_msg
    prediction = "<think>Upon examining the medical image, the small round to oval structures are dispersed throughout the tissue. These structures are relatively uniform in size and shape, which is characteristic of fibroblasts. Fibroblasts are known for their role in producing extracellular matrix and collagen, which can be inferred from the dense, fibrous appearance of the surrounding tissue. Adipocytes typically have a larger, more irregular shape due to lipid content, which is not observed here. Nerve cells would exhibit distinct morphological features such as axons and dendrites, which are absent in this image. Lymphocytes are generally smaller and more densely packed, often seen in clusters, which does not match the observed pattern. Therefore, the morphology and distribution of these cells strongly suggest they are fibroblasts.\n### Final Answer: C. Fibroblasts</think> <answer>C</answer>"
    solution = assistant_msg # Perfect match scenario
    
    # 2. Extract Question for CoTConsistency
    # Usually the question is in the user message.
    # Simple extraction: remove <image> tag
    question = user_msg.replace("<image>", "").strip()

    # 3. Images
    images = data['images']

    # 4. Prepare batch (size 1)
    predictions = [prediction]
    solutions = [solution]
    tasks = ['vqa'] # or 'mcq' depending on needs, but user mentioned VQA metrics like Bleu/Bert
    
    print(f"Input Data:\nQuestion: {question[:100]}...\nPrediction: {prediction[:100]}...\nImages: {images}\n")

    # # --- Test AccuracyBleuReward ---
    # print("\n[1] Testing AccuracyBleuReward...")
    # try:
    #     reward_fn = AccuracyBleuReward()
    #     score = reward_fn(predictions, solutions, task=tasks)
    #     print(f"Score: {score[0]}")
    # except Exception as e:
    #     print(f"Error: {e}")

    # # --- Test AccuracyBertReward ---
    # print("\n[2] Testing AccuracyBertReward...")
    # try:
    #     reward_fn = AccuracyBertReward()
    #     # Mocking or loading actual model? The class loads model on call.
    #     # It might be slow or fail if path is invalid.
    #     score = reward_fn(predictions, solutions, task=tasks)
    #     print(f"Score: {score[0]}")
    # except Exception as e:
    #     print(f"Error: {e}")

    # --- Test CoTConsistencyReward ---
    print("\n[3] Testing CoTConsistencyReward...")
    # Takes 'query' in kwargs
    reward_fn = CoTConsistencyReward()
    score = reward_fn(predictions, solutions, query=[question])
    print(f"Score: {score[0]}")

    # --- Test ConchGLiNERReward ---
    # print("\n[4] Testing ConchGLiNERReward...")
    # reward_fn = ConchGLiNERReward()
    # score = reward_fn(predictions, solutions, images=images)
    # print(f"Score: {score[0]}")

if __name__ == "__main__":
    test_complex_rewards()
