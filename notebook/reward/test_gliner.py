from gliner import GLiNER

model = GLiNER.from_pretrained("/data/ckpt/camembert-bio-gliner-v0.1")

text = """
XiaoMing's age is 20.
"""

labels = ["age"]

entities = model.predict_entities(text, labels, threshold=0.5, flat_ner=True)

for entity in entities:
    print(entity["text"], "=>", entity["label"])