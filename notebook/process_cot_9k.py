import json
import os
import shutil

# Configuration
BASE_DIR = "/data/ljd/VLM-R1/dataset/sft/"
OUTPUT_DIR = "/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/cot/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
INPUT_FILENAME = "pathgen_instruct_close_cot_9144.jsonl"
INPUT_FILEPATH = os.path.join(BASE_DIR, INPUT_FILENAME)

# Templates from merge_sft_datasets.py
COT_QUESTION_TEMPLATE = "{Question}\nThink through the question step by step, enclose your reasoning process in <think>...</think> tags. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags. No extra information or text outside of these tags."
COT_ANSWER_TEMPLATE = "<think>{reasoning}</think> <answer>{answer}</answer>"

def process_close_cot_format(item):
    """
    Processes data in pathgen_instruct_close_cot format.
    """
    question = item["question"].strip()
    formatted_question = COT_QUESTION_TEMPLATE.format(Question=question)

    reasoning = item["reasoning_cot_lingshu32b"]
    answer = item["correct_answer"]
    formatted_answer = COT_ANSWER_TEMPLATE.format(reasoning=reasoning, answer=answer)

    messages = [
        {"role": "user", "content": f"<image>\n{formatted_question}"},
        {"role": "assistant", "content": formatted_answer},
    ]
    return {"messages": messages, "images": [item["image_path"]]}

def main():
    print(f"Checking for input file: {INPUT_FILEPATH}")
    if not os.path.exists(INPUT_FILEPATH):
        print(f"Error: File {INPUT_FILEPATH} not found.")
        return

    # 1. Read and count data
    data = []
    print("Reading data...")
    with open(INPUT_FILEPATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print("Warning: Skipped invalid JSON line.")
    
    count = len(data)
    print(f"Found {count} records.")

    # 2. Rename the original file
    new_source_filename = f"pathgen_instruct_close_cot_{count}.jsonl"
    new_source_filepath = os.path.join(BASE_DIR, new_source_filename)
    
    if INPUT_FILEPATH != new_source_filepath:
        print(f"Renaming {INPUT_FILENAME} to {new_source_filename}...")
        os.rename(INPUT_FILEPATH, new_source_filepath)
    else:
        print("File already has the correct name.")

    # 3. Convert to SFT format
    output_filename = f"pathgen_instruct_close_cot_{count}_sft.jsonl"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
    print(f"Converting to SFT format -> {output_filepath}")

    with open(output_filepath, "w", encoding="utf-8") as outfile:
        for item in data:
            try:
                processed_item = process_close_cot_format(item)
                # Add source field similar to merge script
                processed_item["source"] = "pathgen_instruct_close_cot"
                
                outfile.write(json.dumps(processed_item, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error processing item: {e}")

    print("Done!")

if __name__ == "__main__":
    main()
