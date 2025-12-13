import json
import os
from tqdm import tqdm

def process_item(item):
    """
    Processes an item from the source datasets.
    Expected input format:
    {
        "img": "/path/to/image.jpg",
        "text": "Description...",
        "text_len": ...
    }
    
    Output format:
    {
        "messages": [
            {"role": "assistant", "content": "<image>Description..."}
        ],
        "images": ["/path/to/image.jpg"]
    }
    """
    # The image path is in 'img' key
    image_path = item.get("img")
    
    # The text description is in 'text' key
    text_content = item.get("text", "")
    
    # List of diverse questions for description
    questions = [
        "Describe this image.",
        "What does this image show?",
        "Provide a detailed description of this image.",
        "Explain what is depicted in this pathology image.",
        "Can you describe the content of this image?",
        "What features are visible in this medical image?",
        "Give a description of the tissue shown here.",
        "Write a caption for this image.",
        "What are the key elements in this image?",
        "Detail the visual characteristics of this image.",
        "Please describe this image.",
    ]
    
    selected_question = random.choice(questions)
    
    messages = [
        {"role": "user", "content": f"<image>\n{selected_question}"},
        {"role": "assistant", "content": text_content}
    ]
    
    return {"messages": messages, "images": [image_path]}

import random

def main():
    base_path = "/data/ljd/VLM-R1/dataset/pretrain/"
    output_filename = "pretrain_dataset.jsonl"
    output_sampled_filename = "pretrain_dataset_sampled_0.4.jsonl"
    output_sampled_20k_filename = "pretrain_dataset_sampled_20k.jsonl"
    output_sampled_10_filename = "pretrain_dataset_sampled_0.1.jsonl"
    output_sampled_001_filename = "pretrain_dataset_sampled_0.001.jsonl"
    
    output_filepath = os.path.join(base_path, output_filename)
    output_sampled_filepath = os.path.join(base_path, output_sampled_filename)
    output_sampled_20k_filepath = os.path.join(base_path, output_sampled_20k_filename)
    output_sampled_10_filepath = os.path.join(base_path, output_sampled_10_filename)
    output_sampled_001_filepath = os.path.join(base_path, output_sampled_001_filename)
    
    # List of files to process
    input_files = [
        "pathcap_pair_223169.json",
        "pathgen_pair_1586502.json"
    ]
    
    print(f"Base path: {base_path}")
    print(f"Output file (Full): {output_filepath}")
    print(f"Output file (Sampled 40%): {output_sampled_filepath}")
    print(f"Output file (Sampled 20k): {output_sampled_20k_filepath}")
    print(f"Output file (Sampled 10%): {output_sampled_10_filepath}")
    print(f"Output file (Sampled 0.1%): {output_sampled_001_filepath}")
    
    total_records = 0
    sampled_records = 0
    sampled_20k_records = 0
    sampled_10_records = 0
    sampled_001_records = 0
    
    # Set seed for reproducibility if needed, though pure random is fine for this utility
    random.seed(42)

    with open(output_filepath, "w", encoding="utf-8") as outfile_full, \
         open(output_sampled_filepath, "w", encoding="utf-8") as outfile_sampled, \
         open(output_sampled_20k_filepath, "w", encoding="utf-8") as outfile_sampled_20k, \
         open(output_sampled_10_filepath, "w", encoding="utf-8") as outfile_sampled_10, \
         open(output_sampled_001_filepath, "w", encoding="utf-8") as outfile_sampled_001:
        
        for filename in input_files:
            input_filepath = os.path.join(base_path, filename)
            print(f"\nProcessing {filename}...")
            
            if not os.path.exists(input_filepath):
                print(f"Warning: File not found at {input_filepath}. Skipping.")
                continue
                
            try:
                with open(input_filepath, "r", encoding="utf-8") as infile:
                    # Both input files are JSON lists of objects
                    data = json.load(infile)
                    
                    for item in tqdm(data, desc=f"Converting {filename}"):
                        new_record = process_item(item)
                        
                        # Write to jsonl
                        json_line = json.dumps(new_record, ensure_ascii=False) + "\n"
                        outfile_full.write(json_line)
                        total_records += 1
                        
                        # Generate a random float once for consistency/correlation if desired, or independently.
                        # Using independent samples for simplicity.
                        r = random.random()
                        
                        # Random sampling 40%
                        if r < 0.4:
                            outfile_sampled.write(json_line)
                            sampled_records += 1

                        # Random sampling 20k (Target ~20k out of ~1.8M -> p=0.01105)
                        if r < 0.01105:
                            outfile_sampled_20k.write(json_line)
                            sampled_20k_records += 1
                            
                        # Random sampling 10% (Subset of the 40% implicitly if using same r, which is nice)
                        if r < 0.1:
                            outfile_sampled_10.write(json_line)
                            sampled_10_records += 1

                        # Random sampling 0.1% (Subset of the 10% implicitly)
                        if r < 0.001:
                            outfile_sampled_001.write(json_line)
                            sampled_001_records += 1
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    print(f"\nFinished processing all files.")
    print(f"Total records saved: {total_records}")
    print(f"Sampled 40% records saved: {sampled_records} (approx. {sampled_records/total_records:.2%})")
    print(f"Sampled 20k records saved: {sampled_20k_records} (approx. {sampled_20k_records/total_records:.2%})")
    print(f"Sampled 10% records saved: {sampled_10_records} (approx. {sampled_10_records/total_records:.2%})")
    print(f"Sampled 0.1% records saved: {sampled_001_records} (approx. {sampled_001_records/total_records:.2%})")
    print(f"Full file: {output_filepath}")
    print(f"Sampled 40% file: {output_sampled_filepath}")
    print(f"Sampled 20k file: {output_sampled_20k_filepath}")
    print(f"Sampled 10% file: {output_sampled_10_filepath}")
    print(f"Sampled 0.1% file: {output_sampled_001_filepath}")

if __name__ == "__main__":
    main()
