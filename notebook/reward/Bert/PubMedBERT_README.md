---
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
base_model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
language: en
license: apache-2.0
---

# PubMedBERT 嵌入

这是一个使用 [sentence-transformers](https://www.SBERT.net) 微调的 [PubMedBERT-base](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) 模型。它将句子和段落映射到 768 维的密集向量空间，可用于聚类或语义搜索等任务。训练数据集是使用 [PubMed](https://pubmed.ncbi.nlm.nih.gov/) 标题-摘要对的随机样本以及相似的标题对生成的。

相比于通用模型，PubMedBERT 嵌入在医学文献方面能产生更高质量的嵌入。针对医学子领域进行进一步微调将获得更好的性能。

## 用法 (txtai)

该模型可用于通过 [txtai](https://github.com/neuml/txtai) 构建嵌入数据库，用于语义搜索和/或作为检索增强生成 (RAG) 的知识源。

```python
import txtai

embeddings = txtai.Embeddings(path="neuml/pubmedbert-base-embeddings", content=True)
embeddings.index(documents())

# Run a query
embeddings.search("query to run")
```

## 用法 (Sentence-Transformers)

或者，可以使用 [sentence-transformers](https://www.SBERT.net) 加载该模型。

```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer("neuml/pubmedbert-base-embeddings")
embeddings = model.encode(sentences)
print(embeddings)
```

## 用法 (Hugging Face Transformers)

该模型也可以直接与 Transformers 一起使用。

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Mean Pooling - Take attention mask into account for correct averaging
# 平均池化 - 考虑注意力掩码以进行正确平均
def meanpooling(output, mask):
    embeddings = output[0] # First element of model_output contains all token embeddings
    mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

# Sentences we want sentence embeddings for
# 我们想要获取句子嵌入的句子
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
# 从 HuggingFace Hub 加载模型
tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
model = AutoModel.from_pretrained("neuml/pubmedbert-base-embeddings")

# Tokenize sentences
# 对句子进行分词
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
# 计算 token 嵌入
with torch.no_grad():
    output = model(**inputs)

# Perform pooling. In this case, mean pooling.
# 执行池化。在本例中为平均池化。
embeddings = meanpooling(output, inputs['attention_mask'])

print("Sentence embeddings:")
print(embeddings)
```

## 评估结果

下面展示了该模型与 [MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard) 上顶级基础模型的性能对比。还评估了一个流行的小型模型以及 Hugging Face Hub 上下载量最大的 PubMed 相似度模型。

使用以下数据集评估模型性能。

- [PubMed QA](https://huggingface.co/datasets/qiaojin/PubMedQA)
  - 子集: pqa_labeled, 分割: train, 配对: (question, long_answer)
- [PubMed Subset](https://huggingface.co/datasets/awinml/pubmed_abstract_3_1k)
  - 分割: test, 配对: (title, text)
- [PubMed Summary](https://huggingface.co/datasets/armanc/scientific_papers)
  - 子集: pubmed, 分割: validation, 配对: (article, abstract)

评估结果如下所示。使用 [皮尔逊相关系数](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) 作为评估指标。

| Model                                                                         | PubMed QA | PubMed Subset | PubMed Summary | Average   |
| ----------------------------------------------------------------------------- | --------- | ------------- | -------------- | --------- | 
| [all-MiniLM-L6-v2](https://hf.co/sentence-transformers/all-MiniLM-L6-v2)           | 90.40     | 95.92         | 94.07          | 93.46     |
| [bge-base-en-v1.5](https://hf.co/BAAI/bge-base-en-v1.5)                            | 91.02     | 95.82         | 94.49          | 93.78     |
| [gte-base](https://hf.co/thenlper/gte-base)                                        | 92.97     | 96.90         | 96.24          | 95.37     |
| [**pubmedbert-base-embeddings**](https://hf.co/neuml/pubmedbert-base-embeddings) | **93.27** | **97.00**     | **96.58**      | **95.62** |
| [S-PubMedBert-MS-MARCO](https://hf.co/pritamdeka/S-PubMedBert-MS-MARCO)            | 90.86     | 93.68         | 93.54          | 92.69     |

## 训练

模型使用以下参数进行训练：

**DataLoader**:

`torch.utils.data.dataloader.DataLoader` 长度为 20191，参数如下：
```
{'batch_size': 24, 'sampler': 'torch.utils.data.sampler.RandomSampler', 'batch_sampler': 'torch.utils.data.sampler.BatchSampler'}
```

**Loss**:

`sentence_transformers.losses.MultipleNegativesRankingLoss.MultipleNegativesRankingLoss` 参数如下：
  ```
  {'scale': 20.0, 'similarity_fct': 'cos_sim'}
  ```

fit() 方法的参数：
```
{
    "epochs": 1,
    "evaluation_steps": 500,
    "evaluator": "sentence_transformers.evaluation.EmbeddingSimilarityEvaluator.EmbeddingSimilarityEvaluator",
    "max_grad_norm": 1,
    "optimizer_class": "<class 'torch.optim.adamw.AdamW'>",
    "optimizer_params": {
        "lr": 2e-05
    },
    "scheduler": "WarmupLinear",
    "steps_per_epoch": null,
    "warmup_steps": 10000,
    "weight_decay": 0.01
}
```

## 完整模型架构
```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
)
```

## 更多信息

阅读 [这篇文章](https://medium.com/neuml/embeddings-for-medical-literature-74dae6abf5e0) 了解有关此模型及其构建方式的更多信息。
