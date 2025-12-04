import os
import json
import torch
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.utils import shuffle

# Add pfm to path to import uni modules if needed, but we can use sklearn directly as per request
# The user asked to reference `pfm/uni/downstream/eval_patch_features/linear_probe.py`
# We can implement a simplified version using sklearn directly for simplicity and robustness, 
# or try to import the custom one. Given the context, using sklearn directly is often easier 
# unless the custom one has specific optimizations. 
# The reference file uses sklearn's LogisticRegression inside `_fit_logreg` if `use_sklearn=True`.
# Let's stick to sklearn for simplicity in this notebook script.

def load_data(root_dir, split, model_name):
    feat_path = os.path.join(root_dir, split, "feat", f"{model_name}.pt")
    label_path = os.path.join(root_dir, split, "label", f"{model_name}.json")
    
    if not os.path.exists(feat_path) or not os.path.exists(label_path):
        print(f"Data not found for {model_name} in {split} split.")
        return None, None
        
    print(f"Loading {split} data for {model_name}...")
    features = torch.load(feat_path, map_location="cpu")
    with open(label_path, "r") as f:
        labels = json.load(f)
        
    return features, labels

def train_and_evaluate(model_name, root_dir):
    print(f"\n{'='*20} Processing {model_name} {'='*20}")
    
    # Load Train
    train_feats, train_labels_str = load_data(root_dir, "train", model_name)
    if train_feats is None:
        return
        
    # Load Test
    test_feats, test_labels_str = load_data(root_dir, "test", model_name)
    if test_feats is None:
        return

    # Encode labels
    classes = ["blood", "cancer", "normal", "stroma"]
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    # Filter out unknown labels if any
    def encode(labels_list):
        encoded = []
        valid_indices = []
        for i, label in enumerate(labels_list):
            if label in class_to_idx:
                encoded.append(class_to_idx[label])
                valid_indices.append(i)
            else:
                print(f"Warning: Unknown label '{label}' at index {i}")
        return np.array(encoded), valid_indices

    y_train, train_valid_idx = encode(train_labels_str)
    X_train = train_feats[train_valid_idx].float().numpy()
    
    y_test, test_valid_idx = encode(test_labels_str)
    X_test = test_feats[test_valid_idx].float().numpy()
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Normalization (L2 norm) - often helpful for linear probe on embeddings
    # Reference `linear_probe.py` doesn't explicitly normalize in the snippet, 
    # but often embeddings are normalized. Let's check if they are already normalized?
    # Usually CLIP/VLM features are normalized. If not, sklearn's LogisticRegression 
    # with regularization handles scale, but L2 norm is good practice.
    # Let's skip explicit normalization for now to match raw usage, or maybe normalize?
    # `pfm/chaoyang_zeroshot_classification.py` does `F.normalize(image_features, dim=-1)`.
    # Let's apply L2 normalization.
    from sklearn.preprocessing import normalize
    X_train = normalize(X_train, norm='l2')
    X_test = normalize(X_test, norm='l2')
    
    # Train Logistic Regression
    # C is inverse of regularization strength. 
    # Reference code calculates cost based on feature dim and num classes.
    # cost = (train_feats.shape[1] * NUM_C) / 100
    # Let's use a default or grid search. For simplicity, let's use a standard C=1.0 or the heuristic.
    num_classes = len(classes)
    C_heuristic = (X_train.shape[1] * num_classes) / 100.0
    # C_heuristic might be too large if dim is large (e.g. 4096).
    # Let's stick to a reasonable default or the heuristic if it's standard for this codebase.
    # The reference code uses it, so let's try to use it.
    
    # Explicitly shuffle training data
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
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovo', average='macro')
    except Exception as e:
        print(f"AUC calculation failed: {e}")
        auc = 0.0
        
    print("\nResults:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Save results
    results = {
        "accuracy": acc,
        "f1_weighted": f1,
        "auc": auc,
        "report": classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    }
    
    save_dir = os.path.join(root_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_results.json")
    
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_path}")

def main():
    root_dir = "/data/ljd/Pathology_FM_LLM/expriment/classify/CCRCC/vlm_vision_only"
    
    # Check available models in train/feat
    train_feat_dir = os.path.join(root_dir, "train", "feat")
    if not os.path.exists(train_feat_dir):
        print(f"Train feature directory not found: {train_feat_dir}")
        return
        
    models = [f.replace(".pt", "") for f in os.listdir(train_feat_dir) if f.endswith(".pt")]
    print(f"Found models: {models}")
    
    for model in models:
        train_and_evaluate(model, root_dir)

if __name__ == "__main__":
    main()
