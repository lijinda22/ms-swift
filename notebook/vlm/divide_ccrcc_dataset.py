import os
import json
import random
import glob
from os.path import join as pjoin

def main():
    dataroot = "/data/dataset/classification/CCRCC/tissue_classification/"
    output_dir = "/data/ljd/Pathology_FM_LLM/expriment/classify/CCRCC/"
    
    os.makedirs(output_dir, exist_ok=True)
    
    classes = ["blood", "cancer", "normal", "stroma"]
    
    # Mapping from class folder name to option description
    class_descriptions = {
        "blood": "Red blood cells",
        "cancer": "Renal cancer",
        "normal": "Normal renal",
        "stroma": "Stromal, including smooth muscle, fibrous stroma and blood vessels"
    }
    
    all_samples = []
    
    print(f"Scanning {dataroot}...")
    for cls_name in classes:
        cls_dir = pjoin(dataroot, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"Warning: Directory {cls_dir} not found.")
            continue
            
        images = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
            images.extend(glob.glob(pjoin(cls_dir, ext)))
            images.extend(glob.glob(pjoin(cls_dir, ext.upper())))
            
        print(f"Found {len(images)} images for class '{cls_name}'")
        
        for img_path in images:
            all_samples.append({
                "path": img_path,
                "label": cls_name
            })
            
    if not all_samples:
        print("No images found!")
        return

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_samples)
    
    train_size = int(0.8 * len(all_samples))
    train_data = all_samples[:train_size]
    test_data = all_samples[train_size:]
    
    print(f"Total samples: {len(all_samples)}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    def create_qa_pairs(samples):
        qa_pairs = []
        
        # Fixed options order for consistency in question, but we could shuffle them if needed.
        # For now, let's keep A, B, C, D fixed to the sorted class names or a specific order.
        # The user request implies a fixed set of options in the question text.
        
        # Let's define the options mapping to A, B, C, D
        # We can map the classes list to A, B, C, D
        options_map = {
            "A": classes[0], # blood
            "B": classes[1], # cancer
            "C": classes[2], # normal
            "D": classes[3]  # stroma
        }
        
        # Inverse map for finding the correct answer letter
        class_to_letter = {v: k for k, v in options_map.items()}
        
        question_text = (
            'Which of the following classes does this pathology image belong to: "blood", "cancer", "normal", "stroma"?\n'
            f'A. {class_descriptions[options_map["A"]]}\n'
            f'B. {class_descriptions[options_map["B"]]}\n'
            f'C. {class_descriptions[options_map["C"]]}\n'
            f'D. {class_descriptions[options_map["D"]]}'
        )
        
        for item in samples:
            img_path = item["path"]
            label = item["label"]
            
            answer_letter = class_to_letter[label]
            
            qa_pairs.append({
                "question": question_text,
                "answer": answer_letter,
                "image_path": img_path,
                "gt_label": label # Optional: keep ground truth label for reference
            })
            
        return qa_pairs

    train_qa = create_qa_pairs(train_data)
    test_qa = create_qa_pairs(test_data)
    
    train_json_path = pjoin(output_dir, "train.json")
    test_json_path = pjoin(output_dir, "test.json")
    
    print(f"Saving train data to {train_json_path}...")
    with open(train_json_path, "w") as f:
        json.dump(train_qa, f, indent=4)
        
    print(f"Saving test data to {test_json_path}...")
    with open(test_json_path, "w") as f:
        json.dump(test_qa, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
