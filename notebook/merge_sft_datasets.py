import json
import os
import random
from tqdm import tqdm

# Templates
CLOSE_QUESTION_TEMPLATE = "{Question}\nPlease output only the final answer option directly. Just one letter (A, B, C, or D) with no explanation or additional text."
COT_QUESTION_TEMPLATE = "{Question}\nThink through the question step by step, enclose your reasoning process in <think>...</think> tags. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags. No extra information or text outside of these tags."
COT_ANSWER_TEMPLATE = "<think>{reasoning}</think> <answer>{answer}</answer>"


def process_llava_format(item):
    """
    Processes data in llava/pathgen_instruct format.
    e.g.,
    {
        "image": "path/to/image.jpg",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }
    """
    messages = []
    for conv in item["conversations"]:
        role = "user" if conv["from"] == "human" else "assistant"
        messages.append({"role": role, "content": conv["value"]})

    # Ensure the first user message has <image> token
    if messages and messages[0]["role"] == "user":
        content = messages[0]["content"]
        if "<image>" not in content:
            messages[0]["content"] = "<image>\n" + content
        elif "<image>\n" not in content:
            # If <image> exists but not followed by newline, fix it if needed, 
            # though usually <image> is enough. The user example has <image>\n.
            messages[0]["content"] = content.replace("<image>", "<image>\n", 1)

    return {"messages": messages, "images": [item["image"]]}


def process_desc_format(item):
    """
    Processes data in pathgen_detail_desc format.
    e.g.,
    {
        "image_path": "path/to/image.png",
        "detailed_description_lingshu32b": "..."
    }
    """
    # List of diverse English questions for detailed image description
    desc_questions = [
        "Provide a detailed description of this pathological image.",
        "Can you elaborate on the key features observed in this image?",
        "What are the significant pathological findings in this image?",
        "Describe the cellular morphology and tissue architecture shown here.",
        "What is the overall impression of this histological section?",
        "Walk me through the important details of this image.",
        "Summarize the main characteristics visible in this biopsy.",
        "Offer an in-depth analysis of the image content.",
        "Highlight the most striking aspects of this tissue sample.",
        "What diagnostic clues can be gathered from this image?",
        "Give a comprehensive account of what you see in this microscopic view.",
        "Explain the pathological changes present in this image.",
        "Provide a thorough report on this image.",
        "What abnormalities are evident in this tissue section?",
        "Describe the image from a histopathological perspective.",
        "Detail the micro-anatomical structures and any deviations.",
        "What are the defining features of this image?",
        "Present a full description of the observed pathology.",
        "What is the complete visual assessment of this image?",
    ]

    selected_question = random.choice(desc_questions)

    messages = [
        {"role": "user", "content": f"<image>\n{selected_question}"},
        {"role": "assistant", "content": item["detailed_description_lingshu32b"]},
    ]
    return {"messages": messages, "images": [item["image_path"]]}


def process_close_cot_format(item):
    """
    Processes data in pathgen_instruct_close_cot format.
    e.g.
    {
        "image_path": "...",
        "question": "...",
        "correct_answer": "...",
        "reasoning_cot_lingshu32b": "..."
    }
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


def process_close_subset_format(item):
    """
    Processes data in pathgen_instruct_close_subset format.
    e.g.
    {
        "image": "...",
        "question": "...",
        "answer": "..."
    }
    """
    question = item["question"].strip()
    formatted_question = CLOSE_QUESTION_TEMPLATE.format(Question=question)

    answer = item["answer"].strip()

    messages = [
        {"role": "user", "content": f"<image>\n{formatted_question}"},
        {"role": "assistant", "content": answer},
    ]
    return {"messages": messages, "images": [item["image"]]}


def process_pathmmu_format(item):
    """
    Processes data in pathmmu format.
    e.g.
    {
        "img": "...",
        "question": "...",
        "answer": "..."
    }
    """
    question = item["question"].strip()
    formatted_question = CLOSE_QUESTION_TEMPLATE.format(Question=question)

    answer = item["answer"].strip()

    messages = [
        {"role": "user", "content": f"<image>\n{formatted_question}"},
        {"role": "assistant", "content": answer},
    ]
    return {"messages": messages, "images": [item["img"]]}


def process_pathvqa_format(item):
    """
    Processes data in pathvqa format.
    e.g.
    {
        "image": "...",
        "question": "...",
        "answer": "...",
        "is_YORN": true/false
    }
    """
    question = item["question"].strip()
    answer = item["answer"]

    if item.get("is_YORN", False):
        formatted_question = CLOSE_QUESTION_TEMPLATE.format(Question=question)
    else:
        formatted_question = question

    messages = [
        {"role": "user", "content": f"<image>\n{formatted_question}"},
        {"role": "assistant", "content": answer},
    ]
    return {"messages": messages, "images": [item["image"]]}


def main():
    """
    Main function to process SFT datasets.
    Saves each dataset to a separate file AND merges them into a single file.
    """
    base_path = "/data/ljd/VLM-R1/dataset/sft/"
    output_dir = os.path.join(base_path, "swiftsft_dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    merged_output_file = os.path.join(output_dir, "merged.jsonl")

    # Mapping of filenames to their processing function, source format type, and source name
    # types: "json", "jsonl", "pathmmu_nested"
    datasets_to_process = [
        ("llava_instruct_20000.json", process_llava_format, "json", "llava_instruct"),
        ("pathgen_detail_desc_68362.jsonl", process_desc_format, "jsonl", "pathgen_detail_desc"),
        ("pathgen_instruct_close_cot_39166.jsonl", process_close_cot_format, "jsonl", "pathgen_instruct_close_cot"),
        ("pathgen_instruct_close_subset_96289.jsonl", process_close_subset_format, "jsonl", "pathgen_instruct_close_subset"),
        ("pathgen_instruct_open_102842.json", process_llava_format, "json", "pathgen_instruct_open"),
        ("pathmmu.json", process_pathmmu_format, "pathmmu_nested", "pathmmu"),
        ("pathvqa_eval_pathology.json", process_pathvqa_format, "json", "pathvqa_eval"),
        ("pathvqa_train_pathology.json", process_pathvqa_format, "json", "pathvqa_train"),
    ]

    print(f"Output directory: {output_dir}")
    print(f"Merged output file: {merged_output_file}")

    total_records_all = 0

    with open(merged_output_file, "w", encoding="utf-8") as merged_outfile:
        for filename, processor, file_type, source_name in datasets_to_process:
            input_filepath = os.path.join(base_path, filename)
            output_filename = f"{source_name}.jsonl"
            output_filepath = os.path.join(output_dir, output_filename)
            
            print(f"\nProcessing {filename} -> {output_filename}...")

            if not os.path.exists(input_filepath):
                print(f"Warning: File not found at {input_filepath}. Skipping.")
                continue

            current_file_records = 0
            try:
                with open(input_filepath, "r", encoding="utf-8") as infile, \
                     open(output_filepath, "w", encoding="utf-8") as outfile:
                    
                    if file_type == "json":
                        # Handles files containing a JSON list of objects
                        data = json.load(infile)
                        for item in tqdm(data, desc=f"Converting {filename}"):
                            new_record = processor(item)
                            new_record["source"] = source_name
                            
                            json_line = json.dumps(new_record, ensure_ascii=False) + "\n"
                            outfile.write(json_line)
                            merged_outfile.write(json_line)
                            
                            current_file_records += 1
                    elif file_type == "jsonl":
                        # Handles JSONL files (one JSON object per line)
                        for line in tqdm(infile, desc=f"Converting {filename}"):
                            item = json.loads(line)
                            new_record = processor(item)
                            new_record["source"] = source_name
                            
                            json_line = json.dumps(new_record, ensure_ascii=False) + "\n"
                            outfile.write(json_line)
                            merged_outfile.write(json_line)
                            
                            current_file_records += 1
                    elif file_type == "pathmmu_nested":
                        # Handles pathmmu.json which has nested structure
                        data = json.load(infile)
                        for sub_source_name, source_data in data.items():
                            if "val" in source_data:
                                for item in tqdm(source_data["val"], desc=f"Converting {filename} - {sub_source_name}"):
                                    new_record = processor(item)
                                    # Combine main source name with sub-source (e.g., pathmmu_PubMed)
                                    new_record["source"] = f"{source_name}_{sub_source_name}"
                                    
                                    json_line = json.dumps(new_record, ensure_ascii=False) + "\n"
                                    outfile.write(json_line)
                                    merged_outfile.write(json_line)
                                    
                                    current_file_records += 1
                
                print(f"Saved {current_file_records} records to {output_filepath}")
                total_records_all += current_file_records

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print(f"\nFinished processing datasets.")
    print(f"Total records written: {total_records_all}")
    print(f"Merged file saved at: {merged_output_file}")


if __name__ == "__main__":
    main()