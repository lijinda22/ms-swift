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
    
    # Construct the content string with <image> tag prefix
    # Note: user sample showed "<image>The tissue..." so we concatenate directly
    content = f"<image>{text_content}"
    
    messages = [
        {"role": "assistant", "content": content}
    ]
    
    return {"messages": messages, "images": [image_path]}

import random

def main():
    base_path = "/data/ljd/VLM-R1/dataset/pretrain/"
    output_filename = "pretrain_dataset.jsonl"
    output_sampled_filename = "pretrain_dataset_sampled_0.4.jsonl"
    
    output_filepath = os.path.join(base_path, output_filename)
    output_sampled_filepath = os.path.join(base_path, output_sampled_filename)
    
    # List of files to process
    input_files = [
        "pathcap_pair_223169.json",
        "pathgen_pair_1586502.json"
    ]
    
    print(f"Base path: {base_path}")
    print(f"Output file (Full): {output_filepath}")
    print(f"Output file (Sampled 40%): {output_sampled_filepath}")
    
    total_records = 0
    sampled_records = 0
    
    # Set seed for reproducibility if needed, though pure random is fine for this utility
    random.seed(42)

    with open(output_filepath, "w", encoding="utf-8") as outfile_full, \
         open(output_sampled_filepath, "w", encoding="utf-8") as outfile_sampled:
        
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
                        
                        # Random sampling 40%
                        if random.random() < 0.4:
                            outfile_sampled.write(json_line)
                            sampled_records += 1
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    print(f"\nFinished processing all files.")
    print(f"Total records saved: {total_records}")
    print(f"Sampled records saved: {sampled_records} (approx. {sampled_records/total_records:.2%})")
    print(f"Full file: {output_filepath}")
    print(f"Sampled file: {output_sampled_filepath}")

if __name__ == "__main__":
    main()
