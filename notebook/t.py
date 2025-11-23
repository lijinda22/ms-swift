import json


def process_jsonl(input_file_path: str, output_file_path: str):
    """
    Reads a JSONL file, extracts 'reasoning_cot_lingshu32b' and 'correct_answer',
    combines them into a new 'solution' field with a specific format,
    and writes the modified data to a new JSONL file.

    Args:
        input_file_path: Path to the input JSONL file.
        output_file_path: Path to the output JSONL file.
    """
    processed_data = []
    with open(input_file_path, "r", encoding="utf-8") as infile:
        for i, line in enumerate(infile):
            try:
                item = json.loads(line.strip())
                # Use pop to get the value and remove the key at the same time.
                reasoning = item.pop("reasoning_cot_lingshu32b", None)
                answer = item.pop("correct_answer", None)

                if reasoning is None or answer is None:
                    print(
                        f"Skipping line {i+1} due to missing 'reasoning_cot_lingshu32b' or 'correct_answer'."
                    )
                    continue

                # Construct the solution string
                solution = (
                    f"<think>{reasoning.strip()}</think>"
                    f"<answer>{answer.strip()}</answer>"
                )
                item["solution"] = solution
                processed_data.append(item)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {i+1}: {e} | Line: {line.strip()}")

    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for item in processed_data:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_jsonl_path = "/data/ljd/VLM-R1/dataset/sft/pathgen_cot_26326.jsonl"
    output_jsonl_path = "/data/ljd/VLM-R1/dataset/sft/pathgen_cot_26326_format.jsonl"
    print(f"Processing '{input_jsonl_path}' and saving to '{output_jsonl_path}'...")
    process_jsonl(input_jsonl_path, output_jsonl_path)
    print("Processing complete.")
    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        print("\n--- Content of output.jsonl ---")
        for i, line in enumerate(f):
            if i < 2:  # Print first 2 lines for verification
                print(line.strip())
        print("...")
