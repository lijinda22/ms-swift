import os
import json
import torch
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from os.path import join as j_

# Configuration
DATASETS_BASE_DIR = "/data/ljd/Pathology_FM_LLM/expriment/classify"
DATASETS_CONFIG = [
    {"name": "CCRCC", "path": j_(DATASETS_BASE_DIR, "CCRCC")},
    {"name": "BreaKHis", "path": j_(DATASETS_BASE_DIR, "BreaKHis")},
    {"name": "chaoyang", "path": j_(DATASETS_BASE_DIR, "chaoyang")},
    {"name": "crc100k", "path": j_(DATASETS_BASE_DIR, "crc100k")},
    {"name": "CRC_MSI", "path": j_(DATASETS_BASE_DIR, "CRC_MSI")},
    {"name": "PanCancer-TIL", "path": j_(DATASETS_BASE_DIR, "PanCancer-TIL")},
]

def load_data(root_dir, split, model_name):
    feat_path = j_(root_dir, "vlm_vision_only", split, "feat", f"{model_name}.pt")
    label_path = j_(root_dir, "vlm_vision_only", split, "label", f"{model_name}.json")
    
    if not os.path.exists(feat_path) or not os.path.exists(label_path):
        # Only print if verbose or just return None to skip quietly (or log missing)
        # print(f"Data not found for {model_name} in {split} split at {feat_path}")
        return None, None
        
    print(f"Loading {split} data for {model_name}...")
    try:
        features = torch.load(feat_path, map_location="cpu")
        with open(label_path, "r") as f:
            labels = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
        
    return features, labels

def train_and_evaluate(dataset_name, dataset_path, model_name):
    print(f"\n{'='*20} Processing {dataset_name} - {model_name} {'='*20}")
    
    # Save dir
    save_dir = j_(DATASETS_BASE_DIR, "results_linear_probe", dataset_name, model_name)
    # save_dir = j_(dataset_path, "vlm_vision_only", "results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = j_(save_dir, f"metrics.json")
    
    # if saved model exists, skip
    if os.path.exists(save_path):
        print(f"Metrics already exists for {dataset_name}. Skipping evaluation.")
        return
    
    # Load Train
    train_feats, train_labels_str = load_data(dataset_path, "train", model_name)
    if train_feats is None:
        print(f"Skipping {dataset_name} - {model_name} (Train data missing)")
        return
        
    # Load Test
    test_feats, test_labels_str = load_data(dataset_path, "test", model_name)
    if test_feats is None:
        print(f"Skipping {dataset_name} - {model_name} (Test data missing)")
        return

    # Dynamically determine classes from train set
    unique_labels = sorted(list(set(train_labels_str)))
    if not unique_labels:
        print("No labels found.")
        return
        
    class_to_idx = {cls: i for i, cls in enumerate(unique_labels)}
    print(f"Classes: {unique_labels}")
    
    # Filter and Encode
    def encode(features, labels_list):
        encoded = []
        valid_indices = []
        for i, label in enumerate(labels_list):
            if label in class_to_idx:
                encoded.append(class_to_idx[label])
                valid_indices.append(i)
            # else ignore or warn? For now ignore.
        
        if not valid_indices:
            return None, None
            
        y = np.array(encoded)
        X = features[valid_indices].float().numpy()
        return X, y

    X_train, y_train = encode(train_feats, train_labels_str)
    X_test, y_test = encode(test_feats, test_labels_str)
    
    if X_train is None or X_test is None:
        print("Data empty after filtering.")
        return

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Normalization (L2 norm)
    X_train = normalize(X_train, norm='l2')
    X_test = normalize(X_test, norm='l2')
    
    # Heuristic C
    num_classes = len(unique_labels)
    C_heuristic = (X_train.shape[1] * num_classes) / 100.0
    
    # Shuffle
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    
    print(f"Training Logistic Regression with C={C_heuristic:.4f}...")
    
    clf = LogisticRegression(
        C=C_heuristic,
        max_iter=1000,
        solver='lbfgs',
        multi_class='multinomial',
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    try:
        if num_classes == 2:
             auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
             auc = roc_auc_score(y_test, y_prob, multi_class='ovo', average='macro')
    except Exception as e:
        print(f"AUC calculation failed: {e}")
        auc = 0.0
        
    print("\nResults:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Save results
    results = {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "f1_weighted": f1,
        "auc": auc,
        "report": classification_report(y_test, y_pred, target_names=unique_labels, output_dict=True)
    }
    
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_path}")

def main():
    for ds_conf in DATASETS_CONFIG:
        name = ds_conf["name"]
        path = ds_conf["path"]
        
        # Discover models in train feat dir
        train_feat_dir = j_(path, "vlm_vision_only", "train", "feat")
        if not os.path.exists(train_feat_dir):
            print(f"Dataset {name}: No feature directory found.")
            continue
            
        models = [f.replace(".pt", "") for f in os.listdir(train_feat_dir) if f.endswith(".pt")]
        
        if not models:
             print(f"Dataset {name}: No models found.")
             continue
             
        for model in models:
            train_and_evaluate(name, path, model)

if __name__ == "__main__":
    main()
