import json
import os
import csv
from typing import Dict, List

# ============================================================================
# Configuration (Mirrored from eval_vqammu_vlm.py)
# ============================================================================

MODEL_PATHS = {
    "qwen3_vl-4b-instruct": "/data/ckpt/Qwen3-VL-4B-Instruct/",
    "Lingshu-7B": "/data/ckpt/Lingshu-7B/",
    "qwen3_vl-4b-sft": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_sft_lorarank16/v1-20251222-215426/checkpoint-2922-merged/",
    "qwen3_vl-4b-sft-kd-w0.5_hypocritical": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_sft_kdw0.5_lorarank16_hypocritical/v4-20251224-173742/checkpoint-2922-merged/",
    "qwen3_vl-4b-cpt-sft": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_cpt_sft_lorarank16/v0-20251225-042643/checkpoint-2922-merged/",
    "qwen3_vl-4b-cpt-sft-kd-w0.5_hypocritical": "/data/ljd/Pathology_FM_LLM/expriment/output4paper/sft/qwen3_vl_4b_cpt_sft_kdw0.5_lorarank16_hypocritical/v0-20251227-160912/checkpoint-2922-merged/",
}

# Dataset Constants
PATHMMU_SOURCES = ["PubMed", "EduContent", "PathCLS", "Atlas"]
CLASSIFICATION_DATASETS = ["CCRCC", "BreaKHis", "chaoyang", "crc100k", "CRC_MSI", "PanCancer-TIL"]
CLASSIFICATION_BASE_DIR = "/data/ljd/Pathology_FM_LLM/expriment/classify"
OUTPUT_BASE_DIR = "/data/ljd/Pathology_FM_LLM/expriment/eval_results/sft_test"

DATASET_CONFIGS = {
    "pathvqa_closeset": "mcq",
    "pathvqa_openset": "vqa",
}

# Dynamically add PathMMU datasets
for source in PATHMMU_SOURCES:
    for split in ["val", "test_tiny"]:
        DATASET_CONFIGS[f"pathmmu_{source}_{split}"] = "mcq"

# Dynamically add Classification datasets
for ds in CLASSIFICATION_DATASETS:
    DATASET_CONFIGS[f"classify_{ds}"] = "mcq"

# ============================================================================
# Extraction Logic
# ============================================================================

def main():
    print("Available models:", list(MODEL_PATHS.keys()))
    print("Available datasets:", list(DATASET_CONFIGS.keys()))
    print(f"Output directory: {OUTPUT_BASE_DIR}")

    # Define groups for averaging
    mmu_vqa_datasets = [k for k in DATASET_CONFIGS.keys() if "pathmmu" in k or "pathvqa" in k]
    classification_datasets = [k for k in DATASET_CONFIGS.keys() if "classify" in k]

    summary_data = []

    # Iterate over all defined models
    for model_key in MODEL_PATHS.keys():
        row = {"Model": model_key}
        
        # Track values for averaging
        mmu_vqa_values = []
        classification_values = []

        # Iterate over all defined datasets
        for ds_name, ds_type in DATASET_CONFIGS.items():
            result_file = os.path.join(OUTPUT_BASE_DIR, model_key, f"{ds_name}_results.json")
            
            value = None
            if os.path.exists(result_file):
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        res = json.load(f)
                        metrics = res.get("metrics", {})
                        
                        # Prioritize metric based on dataset type, but fallback if configured otherwise
                        if ds_type == "mcq":
                            # For MCQ, we want Accuracy
                            value = metrics.get("accuracy")
                        else:
                            # For VQA (Open Set), we want BERT Score
                            value = metrics.get("bert_score")
                            
                        # Format if found
                        if value is not None:
                            float_val = float(value)
                            value = f"{float_val:.4f}"
                            
                            # Add to averaging lists
                            if ds_name in mmu_vqa_datasets:
                                mmu_vqa_values.append(float_val)
                            elif ds_name in classification_datasets:
                                classification_values.append(float_val)
                        else:
                            value = "" # Empty if metric not found in JSON
                            
                except Exception as e:
                    print(f"Error reading {result_file}: {e}")
                    value = "Error"
            else:
                value = "" # Empty if file not found
            
            row[ds_name] = value

        # Calculate Means
        if mmu_vqa_values:
            row["Average_VQA_MMU"] = f"{sum(mmu_vqa_values) / len(mmu_vqa_values):.4f}"
        else:
            row["Average_VQA_MMU"] = ""
            
        if classification_values:
            row["Average_Classification"] = f"{sum(classification_values) / len(classification_values):.4f}"
        else:
            row["Average_Classification"] = ""
        
        summary_data.append(row)

    # Write to CSV
    if summary_data:
        # Ensure output directory exists
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        
        csv_file = os.path.join(OUTPUT_BASE_DIR, "evaluation_summary_clean.csv")
        # Define fieldnames: Model column + Averages + all dataset columns
        fieldnames = ["Model", "Average_VQA_MMU", "Average_Classification"] + list(DATASET_CONFIGS.keys())
        
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)
        
        print(f"\nSuccessfully saved summary to: {csv_file}")
        
        # Print a preview
        print("\nPreview:")
        print(f"{'Model':<40} | Avg VQA/MMU | Avg Clf ...")
        print("-" * 80)
        for row in summary_data:
            print(f"{row['Model']:<40} | {row['Average_VQA_MMU']:<11} | {row['Average_Classification']:<7} ...")

if __name__ == "__main__":
    main()
