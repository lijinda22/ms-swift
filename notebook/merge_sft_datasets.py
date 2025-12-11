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



def generate_classification_subset(base_path, sample_size=10000):
    """
    Collects classification datasets, samples 10k items, and saves to a JSONL file in base_path.
    """
    print("Generating classification subset...")
    classify_base_dir = "/data/ljd/Pathology_FM_LLM/expriment/classify"
    datasets = [
        "CCRCC", "BreaKHis", "chaoyang", "crc100k", "CRC_MSI", "PanCancer-TIL"
    ]

    all_items = []

    for ds_name in datasets:
        ds_path = os.path.join(classify_base_dir, ds_name)
        train_file = os.path.join(ds_path, "train.json")
        
        if not os.path.exists(train_file):
            print(f"Warning: {train_file} not found. Skipping.")
            continue
            
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                # Handle varying image path keys
                img_p = item.get("image_path", item.get("path"))
                if not img_p:
                    continue
                    
                if "question" in item and "answer" in item:
                    all_items.append({
                        "image": img_p,
                        "question": item["question"],
                        "answer": item["answer"]
                    })
        except Exception as e:
            print(f"Error reading {train_file}: {e}")

    print(f"Found {len(all_items)} total classification items.")

    if not all_items:
        return None

    # Sample items
    if len(all_items) > sample_size:
        sampled_items = random.sample(all_items, sample_size)
    else:
        sampled_items = all_items
        
    output_filename = f"classification_subset_{len(sampled_items)}.jsonl"
    output_filepath = os.path.join(base_path, output_filename)
    
    print(f"Saving {len(sampled_items)} sampled items to {output_filepath}")
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for item in sampled_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    return output_filename


def main():
    """
    Main function to process SFT datasets.
    Saves each dataset to a separate file AND merges them into a single file.
    """
    base_path = "/data/ljd/VLM-R1/dataset/sft/"
    output_dir = os.path.join(base_path, "swiftsft_dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    merged_temp_file = os.path.join(output_dir, "merged_temp.jsonl")

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

    # Generate and add classification subset
    # Pass base_path so it saves in the same dir as input files
    cls_filename = generate_classification_subset(base_path)
    if cls_filename:
        datasets_to_process.append((cls_filename, process_close_subset_format, "jsonl", "classification_subset"))

    print(f"Output directory: {output_dir}")
    print(f"Temp merged file: {merged_temp_file}")

    total_records_all = 0

    with open(merged_temp_file, "w", encoding="utf-8") as merged_outfile:
        for filename, processor, file_type, source_name in datasets_to_process:
            input_filepath = os.path.join(base_path, filename)
            
            # Temporary output filename, will be renamed after processing to include count
            temp_output_filename = f"{source_name}_temp.jsonl"
            temp_output_filepath = os.path.join(output_dir, temp_output_filename)
            
            print(f"\nProcessing {filename}...")

            if not os.path.exists(input_filepath):
                print(f"Warning: File not found at {input_filepath}. Skipping.")
                continue

            current_file_records = 0
            try:
                with open(input_filepath, "r", encoding="utf-8") as infile, \
                     open(temp_output_filepath, "w", encoding="utf-8") as outfile:
                    
                    if file_type == "json":
                        # Handles files containing a JSON list of objects
                        data = json.load(infile)
                        
                        # Special handling: downsample llava_instruct to 5k
                        if "llava_instruct" in filename:
                             print(f"Downsampling {filename} to 5000 items...")
                             if len(data) > 5000:
                                 data = random.sample(data, 5000)

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
                            try:
                                item = json.loads(line)
                                new_record = processor(item)
                                new_record["source"] = source_name
                                
                                json_line = json.dumps(new_record, ensure_ascii=False) + "\n"
                                outfile.write(json_line)
                                merged_outfile.write(json_line)
                                
                                current_file_records += 1
                            except json.JSONDecodeError:
                                continue
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
                
                # Rename the file to include the count
                final_output_filename = f"{source_name}_{current_file_records}.jsonl"
                final_output_filepath = os.path.join(output_dir, final_output_filename)
                
                # If pathmmu was processed, it might be weird because it has sub-sources but we write to one file "pathmmu_temp".
                # The logic above writes all pathmmu sub-sources to "pathmmu_temp.jsonl".
                # So renaming "pathmmu_temp.jsonl" to "pathmmu_{count}.jsonl" is correct.
                
                if os.path.exists(final_output_filepath):
                    os.remove(final_output_filepath)
                os.rename(temp_output_filepath, final_output_filepath)
                
                print(f"Saved {current_file_records} records to {final_output_filepath}")
                total_records_all += current_file_records

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                if os.path.exists(temp_output_filepath):
                    os.remove(temp_output_filepath)

    # Rename fused merged file
    final_merged_filename = f"merged_{total_records_all}.jsonl"
    final_merged_filepath = os.path.join(output_dir, final_merged_filename)
    
    if os.path.exists(final_merged_filepath):
        os.remove(final_merged_filepath)
    os.rename(merged_temp_file, final_merged_filepath)

    print(f"\nFinished processing datasets.")
    print(f"Total records written: {total_records_all}")
    print(f"Merged file saved at: {final_merged_filepath}")


if __name__ == "__main__":
    main()