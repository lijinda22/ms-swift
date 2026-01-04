
import os
import json
import csv
import argparse

def gather_results(base_path, output_file):
    if not os.path.exists(base_path):
        print(f"Error: Base path not found: {base_path}")
        # Try to suggest check
        if os.name == 'nt' and base_path.startswith('/data'):
             print("Tip: You are on Windows. If '/data' is a mapped drive, ensure it is accessible. If it is on a remote server, run this script there.")
        return

    models_info = {
        "conchv1.5": {"acc": "lin_acc", "auc": "lin_auroc"},
        "uni2": {"acc": "lin_acc", "auc": "lin_auroc"},
        "virchow2": {"acc": "lin_acc", "auc": "lin_auroc"},
        # "qwen3_vl-2b": {"acc": "accuracy", "auc": "auc"},
        # "qwen3_vl-2b-sft": {"acc": "accuracy", "auc": "auc"},
        # "qwen3_vl-2b-sft-kd-w0.5": {"acc": "accuracy", "auc": "auc"},
    }

    try:
        datasets = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    except Exception as e:
        print(f"Error listing directories in {base_path}: {e}")
        return

    # Sort datasets to be consistent
    datasets.sort()
    
    # Check if we have datasets
    if not datasets:
        print(f"No datasets found in {base_path}")
        return

    print(f"Found datasets: {datasets}")

    # Header
    header = ["Dataset"]
    model_names = list(models_info.keys())

    # Add Acc headers
    for model in model_names:
        header.append(f"{model} Acc")
    # Add AUC headers
    for model in model_names:
        header.append(f"{model} AUC")

    rows = []
    for dataset in datasets:
        row = [dataset]
        dataset_res = {}
        
        # Gather data first
        for model in model_names:
            keys = models_info[model]
            metrics_path = os.path.join(base_path, dataset, model, "metrics.json")
            acc = ""
            auc = ""
            
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        raw_acc = data.get(keys['acc'])
                        raw_auc = data.get(keys['auc'])
                        
                        if isinstance(raw_acc, (int, float)):
                            acc = f"{raw_acc:.4f}"
                        else:
                            acc = str(raw_acc) if raw_acc is not None else ""
                            
                        if isinstance(raw_auc, (int, float)):
                            auc = f"{raw_auc:.4f}"
                        else:
                            auc = str(raw_auc) if raw_auc is not None else ""
                            
                except Exception as e:
                    print(f"Error reading {metrics_path}: {e}")
            
            dataset_res[model] = (acc, auc)
        
        # Append Accs
        for model in model_names:
            row.append(dataset_res[model][0])
            
        # Append AUCs
        for model in model_names:
            row.append(dataset_res[model][1])
            
        rows.append(row)

    # Calculate Mean Row
    mean_row = ["Mean"]
    # There are len(header) columns. Column 0 is Dataset.
    num_columns = len(header)
    
    for i in range(1, num_columns):
        values = []
        for row in rows:
            val_str = row[i]
            if val_str:
                try:
                    values.append(float(val_str))
                except ValueError:
                    pass
        
        if values:
            avg = sum(values) / len(values)
            mean_row.append(f"{avg:.4f}")
        else:
            mean_row.append("")
    
    rows.append(mean_row)

    # Print Table (Markdown format)
    # Calculate column widths
    col_widths = [len(h) for h in header]
    for row in rows:
        for i, val in enumerate(row):
            if len(val) > col_widths[i]:
                col_widths[i] = len(val)

    # Create format string
    fmt = "| " + " | ".join([f"{{:<{w}}}" for w in col_widths]) + " |"
    
    print("\nSummary Table:")
    print("-" * (sum(col_widths) + 3 * len(col_widths) + 1))
    print(fmt.format(*header))
    print("| " + " | ".join(["-" * w for w in col_widths]) + " |")
    for row in rows:
        print(fmt.format(*row))
    print("-" * (sum(col_widths) + 3 * len(col_widths) + 1))
    
    if output_file:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"\nResults saved to {output_file}")
        
        # also save as markdown for easy viewing
        md_file = output_file.replace('.csv', '.md')
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"| {' | '.join(header)} |\n")
            f.write(f"| {' | '.join(['---'] * len(header))} |\n")
            for row in rows:
                f.write(f"| {' | '.join(row)} |\n")
        print(f"Markdown table saved to {md_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather linear probe results")
    parser.add_argument("--base_dir", type=str, default="/data/ljd/Pathology_FM_LLM/expriment/classify/results_linear_probe", help="Base directory of results")
    parser.add_argument("--output", type=str, default="/data/ljd/Pathology_FM_LLM/expriment/classify/results_linear_probe/linear_probe_summary.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    gather_results(args.base_dir, args.output)
