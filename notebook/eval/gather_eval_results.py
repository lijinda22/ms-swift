
import os
import json
import csv
import argparse

def gather_vlm_results(base_path, output_file):
    if not os.path.exists(base_path):
        print(f"Error: Base path not found: {base_path}")
        if os.name == 'nt' and base_path.startswith('/data'):
             print("Tip: You are on Windows. If '/data' is a mapped drive, ensure it is accessible. If it is on a remote server, run this script there.")
        return

    # 1. Discover Models (Subdirectories in base_path)
    try:
        found_models = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        
        # User defined preferred order
        preferred_order = [
            "qwen3_vl-2b-instruct",
            "qwen3_vl-2b-sft",
            "qwen3_vl-2b-sft-kd-w0.5",
            "qwen3_vl-2b-cpt_sft"
        ]
        
        # Filter found models to only include those in preferred order (or append others? Request implies "listing" specific ones)
        # We will prioritize preferred ones, and append any others found at the end sorted alphabetically, 
        # OR strict filtering if the user implies ONLY these. "列改为按照..." usually means show these. 
        # Let's show these primarily.
        
        models = []
        for pm in preferred_order:
            if pm in found_models:
                models.append(pm)
            else:
                # If a preferred model is missing, should we include it as empty column? 
                # Usually better to see it's missing (empty column) than not seeing it.
                # But logic downstream relies on directory existence for discover... 
                # Actually downstream `os.path.exists(json_path)` handles missing checks.
                # So let's force the list.
                models.append(pm)
        
        # Add any other models found that were not in preferred list?
        # User request seems specific. Let's stick to the preferred list to keep table clean as requested.
        # If user wants others, they would ask.
        
    except Exception as e:
        print(f"Error listing directories in {base_path}: {e}")
        return

    # Check if we have effectively found any (strictly speaking we forced the list, so 'models' is not empty)
    # But let's check if we found *any* of them in reality to warn?
    # existing_models = [m for m in models if m in found_models]
    # if not existing_models:
    #      print(f"Warning: None of the requested models found in {base_path}. listing: {found_models}")

    print(f"Target models: {models}")

    # 2. Discover Datasets & Metrics across all models
    # We want to enforce a specific order: PathVQA -> PathMMU -> Classification
    # We will verify existence of files to confirm dataset presence.
    
    # Expected groups
    pathvqa_datasets = ["pathvqa_closeset", "pathvqa_openset"]
    
    # PathMMU - we look for anything starting with pathmmu_
    pathmmu_datasets = set()
    
    # Classification - fixed list
    classification_datasets = [
        "CCRCC", "BreaKHis", "chaoyang", "crc100k", "CRC_MSI", "PanCancer-TIL"
    ]
    
    # Scan for potential datasets (especially PathMMU which might vary)
    all_found_datasets = set()
    for model in models:
        model_dir = os.path.join(base_path, model)
        files = os.listdir(model_dir)
        for f in files:
            if f.endswith("_results_vllm.json"):
                ds_name = f.replace("_results_vllm.json", "")
                all_found_datasets.add(ds_name)
                if ds_name.startswith("pathmmu_"):
                    pathmmu_datasets.add(ds_name)

    # Convert to list and sort
    pathmmu_datasets = sorted(list(pathmmu_datasets))
    
    # Final Ordered List logic
    final_datasets = []
    
    # 2.1 Add PathVQA
    for ds in pathvqa_datasets:
        if ds in all_found_datasets:
            final_datasets.append(ds)
        else:
             # Just in case we want to show empty rows for missing expected datasets? 
             # Let's include them to be explicit, but check if they exist in all_found_datasets involves logic.
             # User asked for "16 datasets", so let's stick to what we found but ordered.
             pass

    # 2.2 Add PathMMU
    final_datasets.extend(pathmmu_datasets)
    
    # 2.3 Add Classification
    for ds in classification_datasets:
        if ds in all_found_datasets:
            final_datasets.append(ds)

    # 3. Build Table
    # Rows: Datasets
    # Columns: Models
    # Cells: Metrics
    
    header = ["Dataset"] + models
    rows = []

    for ds in final_datasets:
        row = [ds]
        for model in models:
            json_path = os.path.join(base_path, model, f"{ds}_results_vllm.json")
            cell_value = ""
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        metrics = data.get("metrics", {})
                        
                        # Determine metrics to show
                        # Type hints usually exist but based on name:
                        if "pathvqa_closeset" in ds or ds in classification_datasets or "mcq" in data.get("dataset_type", ""):
                            # MCQ -> Accuracy
                            acc = metrics.get("accuracy")
                            if acc is None:
                                # fallback or check if it is openset logic
                                if "exact_match" in metrics:
                                     # Sometimes mislabeled?
                                     acc = metrics.get("exact_match")
                            
                            if isinstance(acc, (int, float)):
                                cell_value = f"{acc:.4f}"
                            else:
                                cell_value = str(acc) if acc is not None else "N/A"
                                
                        elif "pathvqa_openset" in ds or "vqa" in data.get("dataset_type", ""):
                            # VQA -> EM / BLEU4
                            em = metrics.get("exact_match")
                            bleu = metrics.get("bleu4")
                            
                            em_str = f"{em:.4f}" if isinstance(em, (int, float)) else "N/A"
                            bleu_str = f"{bleu:.4f}" if isinstance(bleu, (int, float)) else "N/A"
                            
                            # cell_value = f"{em_str} / {bleu_str}"
                            cell_value = f"{bleu_str}"
                        else:
                            # Fallback dump
                            cell_value = str(metrics)
                            
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")
                    cell_value = "Err"
            else:
                 cell_value = "-"
            
            row.append(cell_value)
        rows.append(row)

    # 4. Output
    # Calculate column widths for display
    col_widths = [len(str(h)) for h in header]
    for row in rows:
        for i, val in enumerate(row):
            if len(str(val)) > col_widths[i]:
                col_widths[i] = len(str(val))

    # Create format string
    fmt = "| " + " | ".join([f"{{:<{w}}}" for w in col_widths]) + " |"
    
    print("\nVLM Evaluation Results Summary:")
    print("-" * (sum(col_widths) + 3 * len(col_widths) + 1))
    print(fmt.format(*header))
    print("| " + " | ".join(["-" * w for w in col_widths]) + " |")
    for row in rows:
        print(fmt.format(*row))
    print("-" * (sum(col_widths) + 3 * len(col_widths) + 1))
    
    if output_file:
        # Save CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"\nResults saved to {output_file}")
        
        # Save Markdown
        md_file = output_file.replace('.csv', '.md')
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"| {' | '.join(header)} |\n")
            f.write(f"| {' | '.join(['---'] * len(header))} |\n")
            for row in rows:
                f.write(f"| {' | '.join(row)} |\n")
        print(f"Markdown table saved to {md_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather VLM evaluation results")
    parser.add_argument("--base_dir", type=str, default="/data/ljd/Pathology_FM_LLM/expriment/eval_results", help="Base directory of results")
    parser.add_argument("--output", type=str, default="/data/ljd/Pathology_FM_LLM/expriment/eval_results/summary.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    gather_vlm_results(args.base_dir, args.output)
