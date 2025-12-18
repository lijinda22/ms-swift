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


def process_close_question_format(item):
    """
    Processes data in pathgen_instruct_close_subset format or classification format.
    Applies CLOSE_QUESTION_TEMPLATE to the question.
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



def generate_classification_subset(base_path, sample_size=100000):
    """
    Collects classification datasets, samples 10k items, and saves to a JSONL file in base_path.
    The output items contain raw 'question'. The 'process_close_question_format' function
    will be used later to apply CLOSE_QUESTION_TEMPLATE.
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
    Saves each dataset group to separate files.
    Does NOT merge everything into a single file anymore.
    """
    base_path = "/data/ljd/VLM-R1/dataset/sft/"
    output_dir = os.path.join(base_path, "swiftsft_dataset_new")
    os.makedirs(output_dir, exist_ok=True)
    
    # Group 1: Datasets to be merged into 'pathgen_vqa'
    pathgen_vqa_sources = [
        ("llava_instruct_20000.json", process_llava_format, "json", "llava_instruct"),
        ("pathgen_detail_desc_68362.jsonl", process_desc_format, "jsonl", "pathgen_detail_desc"),
        ("pathgen_instruct_close_cot_39166.jsonl", process_close_cot_format, "jsonl", "pathgen_instruct_close_cot"),
        ("pathgen_instruct_close_subset_96289.jsonl", process_close_question_format, "jsonl", "pathgen_instruct_close_subset"),
        ("pathgen_instruct_open_102842.json", process_llava_format, "json", "pathgen_instruct_open"),
    ]

    # Group 2: Independent datasets (processed separately)
    independent_sources = [
        ("pathmmu.json", process_pathmmu_format, "pathmmu_nested", "pathmmu"),
        ("pathvqa_eval_pathology.json", process_pathvqa_format, "json", "pathvqa_eval"),
        ("pathvqa_train_pathology.json", process_pathvqa_format, "json", "pathvqa_train"),
        ("pathvqa_test_pathology.json", process_pathvqa_format, "json", "pathvqa_test"),
    ]

    # Generate and add classification subset to independent sources
    cls_filename = generate_classification_subset(base_path)
    if cls_filename:
        independent_sources.append((cls_filename, process_close_question_format, "jsonl", "classification_subset"))

    print(f"Output directory: {output_dir}")
    
    total_records_all = 0

    # --- Process Group 1: PathGen VQA (Merged) ---
    print("\n=== Processing PathGen VQA Group ===")
    pathgen_vqa_temp = os.path.join(output_dir, "pathgen_vqa_temp.jsonl")
    pathgen_vqa_count = 0
    
    with open(pathgen_vqa_temp, "w", encoding="utf-8") as pathgen_outfile:
        for filename, processor, file_type, source_name in pathgen_vqa_sources:
            input_filepath = os.path.join(base_path, filename)
            print(f"Processing {filename} for pathgen_vqa...")

            if not os.path.exists(input_filepath):
                print(f"Warning: File not found at {input_filepath}. Skipping.")
                continue

            try:
                with open(input_filepath, "r", encoding="utf-8") as infile:
                    if file_type == "json":
                        data = json.load(infile)
                        # Special handling: downsample llava_instruct to 5k
                        if "llava_instruct" in filename:
                                print(f"Downsampling {filename} to 5000 items...")
                                if len(data) > 5000:
                                    data = random.sample(data, 5000)

                        for item in tqdm(data, desc=f"Converting {filename}"):
                            new_record = processor(item)
                            new_record["source"] = source_name # Keep original source name in record
                            
                            json_line = json.dumps(new_record, ensure_ascii=False) + "\n"
                            pathgen_outfile.write(json_line)
                            pathgen_vqa_count += 1
                    
                    elif file_type == "jsonl":
                        for line in tqdm(infile, desc=f"Converting {filename}"):
                            try:
                                item = json.loads(line)
                                new_record = processor(item)
                                new_record["source"] = source_name
                                
                                json_line = json.dumps(new_record, ensure_ascii=False) + "\n"
                                pathgen_outfile.write(json_line)
                                pathgen_vqa_count += 1
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                    print(f"Error processing file {filename}: {e}")

    # Rename pathgen_vqa file
    final_pathgen_name = f"pathgen_vqa_{pathgen_vqa_count}.jsonl"
    final_pathgen_path = os.path.join(output_dir, final_pathgen_name)
    if os.path.exists(final_pathgen_path):
        os.remove(final_pathgen_path)
    os.rename(pathgen_vqa_temp, final_pathgen_path)
    print(f"Saved {pathgen_vqa_count} records to {final_pathgen_path}")
    total_records_all += pathgen_vqa_count


    # --- Process Group 2: Independent Sources ---
    print("\n=== Processing Independent Sources ===")
    for filename, processor, file_type, source_name in independent_sources:
        input_filepath = os.path.join(base_path, filename)
        
        # Temporary output filename
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
                    data = json.load(infile)
                    for item in tqdm(data, desc=f"Converting {filename}"):
                        new_record = processor(item)
                        new_record["source"] = source_name
                        
                        json_line = json.dumps(new_record, ensure_ascii=False) + "\n"
                        outfile.write(json_line)
                        current_file_records += 1
                        
                elif file_type == "jsonl":
                    for line in tqdm(infile, desc=f"Converting {filename}"):
                        try:
                            item = json.loads(line)
                            new_record = processor(item)
                            new_record["source"] = source_name
                            
                            json_line = json.dumps(new_record, ensure_ascii=False) + "\n"
                            outfile.write(json_line)
                            current_file_records += 1
                        except json.JSONDecodeError:
                            continue

                elif file_type == "pathmmu_nested":
                    # Handles pathmmu.json which has nested structure
                    data = json.load(infile)
                    # Switch source_name to pathmmu_val for the final filename renaming
                    source_name = "pathmmu_val"

                    # Extra writers for test/test_tiny
                    pathmmu_splits_extra = ['test', 'test_tiny']
                    extra_writers = {}
                    extra_counts = {s: 0 for s in pathmmu_splits_extra}
                    extra_temp_paths = {}
                    
                    for split in pathmmu_splits_extra:
                        p = os.path.join(output_dir, f"pathmmu_{split}_temp.jsonl")
                        extra_temp_paths[split] = p
                        extra_writers[split] = open(p, "w", encoding="utf-8")

                    try:
                        for sub_source_name, source_data in data.items():
                            # 1. Process VAL (Write to main outfile)
                            if "val" in source_data:
                                for item in tqdm(source_data["val"], desc=f"Converting {filename} - {sub_source_name} (val)"):
                                    new_record = processor(item)
                                    new_record["source"] = f"pathmmu_{sub_source_name}"
                                    
                                    json_line = json.dumps(new_record, ensure_ascii=False) + "\n"
                                    outfile.write(json_line)
                                    # merged_outfile removed
                                    
                                    current_file_records += 1

                            # 2. Process TEST / TEST_TINY (Separate files only)
                            for split in pathmmu_splits_extra:
                                if split in source_data:
                                    for item in tqdm(source_data[split], desc=f"Converting {filename} - {sub_source_name} ({split})"):
                                        new_record = processor(item)
                                        # Same here
                                        new_record["source"] = f"pathmmu_{sub_source_name}"

                                        json_line = json.dumps(new_record, ensure_ascii=False) + "\n"
                                        extra_writers[split].write(json_line)
                                        extra_counts[split] += 1
                    finally:
                        for w in extra_writers.values():
                            w.close()

                    # Rename extra split files
                    for split in pathmmu_splits_extra:
                            count = extra_counts[split]
                            temp_p = extra_temp_paths[split]
                            if os.path.exists(temp_p):
                                final_p = os.path.join(output_dir, f"pathmmu_{split}_{count}.jsonl")
                                if os.path.exists(final_p):
                                    os.remove(final_p)
                                os.rename(temp_p, final_p)
                                print(f"Saved {count} records to {final_p}")
            
            # Rename the file to include the count
            final_output_filename = f"{source_name}_{current_file_records}.jsonl"
            final_output_filepath = os.path.join(output_dir, final_output_filename)
            
            if os.path.exists(final_output_filepath):
                os.remove(final_output_filepath)
            os.rename(temp_output_filepath, final_output_filepath)
            
            print(f"Saved {current_file_records} records to {final_output_filepath}")
            total_records_all += current_file_records

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            if os.path.exists(temp_output_filepath):
                os.remove(temp_output_filepath)


    print(f"\nFinished processing datasets.")
    print(f"Total records processed: {total_records_all}")


if __name__ == "__main__":
    main()
