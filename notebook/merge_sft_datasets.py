import json
import os
from tqdm import tqdm


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

    # Ensure the first user message has <image> token and it's structured properly
    if messages and messages[0]["role"] == "user":
        content = messages[0]["content"]
        if "<image>" not in content:
            messages[0]["content"] = "<image>\n" + content
        elif "<image>\n" not in content:
            messages[0]["content"] = content.replace("<image>", "<image>\n", 1)

    return {"messages": messages, "images": [item["image"]]}


def process_cot_format(item):
    """
    Processes data in pathgen_cot format.
    e.g.,
    {
        "image_path": "path/to/image.png",
        "question": "...",
        "solution": "<think>...</think><answer>...</answer>"
    }
    """
    question = item["question"].strip()
    user_content = f"<image>\n{question}"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": item["solution"]},
    ]
    return {"messages": messages, "images": [item["image_path"]]}


import random


def process_desc_format(item):
    """
    Processes data in pathgen_detail_desc format.
    e.g.,
    {
        "image_path": "path/to/image.png",
        "detailed_description_lingshu32b": "..."
    }
    """
    # List of 20 diverse English questions for detailed image description
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


def main():
    """
    Main function to merge SFT datasets into the specified format.
    """
    base_path = "/data/ljd/VLM-R1/dataset/sft/"
    output_file = os.path.join(base_path, "merged_sft_dataset.jsonl")

    # Mapping of filenames to their processing function and source format type
    datasets_to_process = {
        "llava_instruct_20000.json": (process_llava_format, "json"),
        "pathgen_instruct_open_102842.json": (process_llava_format, "json"),
        "pathgen_cot_26326_format.jsonl": (process_cot_format, "jsonl"),
        "pathgen_detail_desc_68362.jsonl": (process_desc_format, "jsonl"),
    }

    print(f"Output will be saved to: {output_file}")

    total_records = 0
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename, (processor, file_type) in datasets_to_process.items():
            filepath = os.path.join(base_path, filename)
            print(f"\nProcessing {filename}...")

            if not os.path.exists(filepath):
                print(f"Warning: File not found at {filepath}. Skipping.")
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as infile:
                    if file_type == "json":
                        # Handles files containing a JSON list of objects
                        data = json.load(infile)
                        for item in tqdm(data, desc=f"Converting {filename}"):
                            new_record = processor(item)
                            outfile.write(
                                json.dumps(new_record, ensure_ascii=False) + "\n"
                            )
                            total_records += 1
                    elif file_type == "jsonl":
                        # Handles JSONL files (one JSON object per line)
                        for line in tqdm(infile, desc=f"Converting {filename}"):
                            item = json.loads(line)
                            new_record = processor(item)
                            outfile.write(
                                json.dumps(new_record, ensure_ascii=False) + "\n"
                            )
                            total_records += 1
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print(f"\nFinished merging datasets.")
    print(f"Total records written: {total_records}")
    print(f"Merged file saved at: {output_file}")


if __name__ == "__main__":
    main()
