from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Mean Pooling - Take attention mask into account for correct averaging
def meanpooling(output, mask):
    embeddings = output[0] # First element of model_output contains all token embeddings
    mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

def calculate_similarity(text1, text2, model_name="/data/ckpt/pubmedbert-base-embeddings/"):
    """
    Calculate cosine similarity between two strings using PubMedBERT.
    """
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    sentences = [text1, text2]
    
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling
    sentence_embeddings = meanpooling(model_output, encoded_input['attention_mask'])
    
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    # Calculate cosine similarity
    similarity = F.cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0))
    
    return similarity.item()

if __name__ == "__main__":
    str1 = "This is an example sentence"
    str2 = "Each sentence is converted"
    
    print(f"String 1: {str1}")
    print(f"String 2: {str2}")
    
    score = calculate_similarity(str1, str2)
    print(f"Similarity Score: {score:.4f}")
    
    # Example with similar medical text
    str3 = "The patient has a fever."
    str4 = "The patient presents with elevated temperature."
    
    
    print(f"\nString 3: {str3}")
    print(f"String 4: {str4}")
    
    score_med = calculate_similarity(str3, str4)
    print(f"Similarity Score: {score_med:.4f}")
