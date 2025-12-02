import os

def rename_jsonl_with_length(file_path):
    """
    Renames a .jsonl file by appending its line count (length) to the filename.
    e.g., data.jsonl -> data_1000.jsonl
    """
    if not os.path.exists(file_path):
        print(f"Skipping: {file_path} (File not found)")
        return

    try:
        # Count lines (assuming one JSON object per line)
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        
        # Construct new filename
        directory, filename = os.path.split(file_path)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{line_count}{ext}"
        new_file_path = os.path.join(directory, new_filename)
        
        # Rename
        os.rename(file_path, new_file_path)
        print(f"Renamed: {filename} -> {new_filename}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # Paths matched to the output files in cot_data_generator.py
    target_files = [
        "/data/ljd/VLM-R1/dataset/sft/pathgen_instruct_close_cot.jsonl",
        "/data/ljd/VLM-R1/dataset/sft/pathgen_instruct_close_subset.jsonl"
    ]
    
    print("Starting renaming process...")
    for file_path in target_files:
        rename_jsonl_with_length(file_path)
    print("Done.")
