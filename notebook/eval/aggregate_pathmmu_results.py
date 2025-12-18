#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate PathMMU per-source results into a summary table.
"""

import json
import os
from typing import Dict, List

EVAL_RESULTS_DIR = "/data/ljd/Pathology_FM_LLM/expriment/eval_results"
PATHMMU_SOURCES = ["PubMed", "SocialPath", "EduContent", "PathCLS", "Atlas"]

def aggregate_pathmmu_results(model_name: str) -> Dict:
    """Aggregate PathMMU results for a specific model."""
    model_dir = os.path.join(EVAL_RESULTS_DIR, model_name)
    
    if not os.path.exists(model_dir):
        print(f"No results found for {model_name}")
        return {}
    
    results = {
        "model": model_name,
        "test": {},
        "test_tiny": {}
    }
    
    for source in PATHMMU_SOURCES:
        for split in ["test", "test_tiny"]:
            result_file = os.path.join(model_dir, f"pathmmu_{source}_{split}_results.json")
            
            if os.path.exists(result_file):
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                results[split][source] = {
                    "accuracy": data["metrics"]["accuracy"],
                    "num_samples": data["num_samples"]
                }
            else:
                results[split][source] = {
                    "accuracy": None,
                    "num_samples": 0
                }
    
    return results


def print_summary_table(results: Dict):
    """Print a formatted summary table."""
    model_name = results["model"]
    
    print(f"\n{'='*80}")
    print(f"PathMMU Results Summary - {model_name}")
    print(f"{'='*80}\n")
    
    for split in ["test", "test_tiny"]:
        print(f"\n{split.upper()}:")
        print(f"{'-'*60}")
        print(f"{'Source':<20} {'Accuracy':<15} {'Samples':<10}")
        print(f"{'-'*60}")
        
        total_correct = 0
        total_samples = 0
        
        for source in PATHMMU_SOURCES:
            data = results[split].get(source, {})
            acc = data.get("accuracy")
            num = data.get("num_samples", 0)
            
            if acc is not None:
                acc_str = f"{acc:.4f}"
                total_correct += int(acc * num)
                total_samples += num
            else:
                acc_str = "N/A"
            
            print(f"{source:<20} {acc_str:<15} {num:<10}")
        
        print(f"{'-'*60}")
        if total_samples > 0:
            overall_acc = total_correct / total_samples
            print(f"{'OVERALL':<20} {overall_acc:.4f}{'':<10} {total_samples:<10}")
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate PathMMU per-source results")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    
    args = parser.parse_args()
    
    results = aggregate_pathmmu_results(args.model)
    
    if not results:
        return
    
    print_summary_table(results)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
