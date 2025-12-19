from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba

def calculate_bleu4_old(reference, hypothesis):
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference.lower().split()], hypothesis.lower().split(), smoothing_function=smoothing)

def calculate_bleu4_new(reference, hypothesis):
    hyp_tokens = list(jieba.cut(hypothesis))
    ref_tokens = list(jieba.cut(reference))
    if not hyp_tokens or not ref_tokens:
        return 0.0
    smoothing = SmoothingFunction().method3
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)

# Test cases
test_cases = [
    ("This is a test.", "This is a test."),
    ("这是一次测试。", "这是一次测试。"),
    ("The patient has a tumor.", "A tumor is seen in the patient."),
    ("医生说病人有肿瘤。", "病人被诊断出患有肿瘤。"),
]

for ref, hyp in test_cases:
    print(f"Ref: {ref}")
    print(f"Hyp: {hyp}")
    print(f"Old BLEU: {calculate_bleu4_old(ref, hyp):.4f}")
    print(f"New BLEU: {calculate_bleu4_new(ref, hyp):.4f}")
    print("-" * 20)
