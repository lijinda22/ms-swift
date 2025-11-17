# {
#   "PubMed": {
#     "val": [{},{}, ...]
#     "test": [{},{}, ...]
#     "test_tiny": [{},{}, ...]
#   }
#   ....
# }
#   {
#     "No": 0,
#     "img": "c39fcdb66d3407140c9e5297b46f61182f40fa3fa6a7bb9c0c822b759ce94f7f.png",
#     "question": "Based on the cellular morphology observed in the image, which feature is consistent with the histopathological appearance?",
#     "options": [
#       "A) Monotonous cell population with uniform nuclei",
#       "B) Prominent areas of necrosis and hemorrhage",
#       "C) High mitotic index with atypical mitotic figures",
#       "D) Marked pleomorphism and anaplasia"
#     ],
#     "answer": "A) Monotonous cell population with uniform nuclei",
#     "explanation": "The image shows a proliferation of ovoid to polygonal cells with eosinophilic cytoplasm that are uniform in size and shape, and where the nuclei appear round to oval with no overt cytological atypia. This is consistent with a monotonous appearance of cells, making option A the best answer. Options A and B are incorrect because no mitotic figures or marked pleomorphism are observed in the image. Option B is also incorrect as there are no prominent areas of necrosis or hemorrhage visible in the field."
#   },
#  image_folder = /data/ckpt/PathMMU/images/
# 实现代码, 检查每个图像路径是否有效, img_path = f"{image_folder}/{img}"
# 其中 image_folder 是存放图像的文件夹路径, img 是图像文件名
# 如果无效, 记录 前两级的key, 比如: PubMed, val

import os
import json
import re

dirpath = "/data/dataset/vqa/PathMMU/"
json_path = f"{dirpath}/data.json"
image_folder = f"{dirpath}/images"

with open(json_path, "r") as f:
    data = json.load(f)

invalid_images = set()

for key, value in data.items():
    for key2, values2 in value.items():
        # key1: PubMed. key2: val
        for item in values2:
            img = item.get("img")
            img_path = os.path.join(image_folder, img)
            if not os.path.exists(img_path):
                invalid_images.add((key, key2))

print("Invalid image paths found in the following datasets:")
for key, key2 in invalid_images:
    print(f" - {key} / {key2}")


new_data = {}
for key, value in data.items():
    for key2, values2 in value.items():
        for item in values2:
            question = item["question"] + " " + " ".join(item["options"])
            answer = item["answer"]
            img = item.get("img")
            img_path = os.path.join(image_folder, img)
            if not os.path.exists(img_path):
                continue
            assert re.match(r"^[A-Z]", answer), f"Invalid answer:{answer}, {question}"
            answer = answer.split(")")[0]
            new_item = {"answer": answer, "question": question, "image": item["img"]}
            new_data.setdefault(key, {}).setdefault(key2, []).append(new_item)

save_path = f"{dirpath}/data_new.json"
with open(save_path, "w") as f:
    json.dump(new_data, f, indent=4)
