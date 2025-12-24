
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def analyze_entailment(model_path, pairs):
    """
    Load model and analyze a list of premise-hypothesis pairs with expected results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path}...")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    model.eval()

    label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    # Map user expectation strings to model labels if needed, or just use standard english
    
    print("\n" + "="*180)
    print(f"{'Premise':<50} | {'Hypothesis':<50} | {'Pred':<10} | {'Expect':<10} | {'Match':<5} | {'Probs (E/N/C)'}")
    print("-" * 180)

    for premise, hypothesis, expected_label in pairs:
        inputs = tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation='only_first').to(device)
        
        with torch.no_grad():
            logits = model(inputs)[0]
            probs = logits.softmax(dim=1).cpu().numpy()[0]
        
        # Labels: 0=entailment, 1=neutral, 2=contradiction
        predicted_index = np.argmax(probs)
        predicted_label = label_map[predicted_index]
        
        # Check consistency
        is_consistent = (predicted_label == expected_label)
        match_str = "YES" if is_consistent else "NO"
        
        # Truncate for display
        p_disp = (premise[:47] + '...') if len(premise) > 47 else premise
        h_disp = (hypothesis[:47] + '...') if len(hypothesis) > 47 else hypothesis
        
        # Format probs
        probs_str = f"{probs[0]:.3f}/{probs[1]:.3f}/{probs[2]:.3f}"
        
        print(f"{p_disp:<50} | {h_disp:<50} | {predicted_label:<10} | {expected_label:<10} | {match_str:<5} | {probs_str}")

if __name__ == "__main__":
    # Model path on local machine
    model_path = "/data/ckpt/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-mnli/"
    
    # Define groups of comparative data for analysis
    # Format: (Premise, Hypothesis, Expected Label)
    # Expected Labels: 'entailment', 'neutral', 'contradiction'
    comparison_groups = [
        # Group 1: Expression Levels
        ('EpCAM is overexpressed in breast cancer', 'EpCAM is downregulated in breast cancer.', 'contradiction'),
        ('EpCAM is overexpressed in breast cancer', 'High levels of EpCAM are detected in breast cancer.', 'entailment'),
        ('EpCAM is overexpressed in breast cancer', 'EpCAM levels are unaffected in breast cancer.', 'contradiction'),

        # Group 2: Drug Efficiency
        ('The new inhibitor reduced tumor volume by 50% in the study group.', 'The inhibitor was effective in reducing tumor size.', 'entailment'),
        ('The new inhibitor reduced tumor volume by 50% in the study group.', 'The inhibitor had no impact on tumor growth.', 'contradiction'),
        ('The new inhibitor reduced tumor volume by 50% in the study group.', 'The control group showed a 50% reduction in tumor volume.', 'neutral'),

        # Group 3: Disease Association
        ('Hypertension is a risk factor for cardiovascular disease.', 'High blood pressure increases the risk of heart disease.', 'entailment'),
        ('Hypertension is a risk factor for cardiovascular disease.', 'Hypertension protects against cardiovascular events.', 'contradiction'),
        
        # Group 4: Pathological Findings
        ('Histology revealed clear cell renal cell carcinoma.', 'The tissue sample shows characteristics of kidney cancer.', 'entailment'),
        ('Histology revealed clear cell renal cell carcinoma.', 'The biopsy was negative for malignancy.', 'contradiction'),
    ]
    
    print(f"Starting analysis with {len(comparison_groups)} pairs...")
    analyze_entailment(model_path, comparison_groups)
