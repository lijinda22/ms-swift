from PIL import Image
import os
import shutil
import json


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data


def convert_tiff_to_jpeg(tiff_file_path, jpeg_file_path):
    with Image.open(tiff_file_path) as img:
        img = img.convert('RGB')
        img.save(jpeg_file_path, 'JPEG')


# pathcls_img_dir = '/data/dataset/vqa/PathMMU/PathMMU_add/'
# data = load_data('/data/dataset/vqa/PathMMU/data.json')
# data_pathcls = data['Atlas']
# # data_pathcls = data['PathCLS']

# for subset, subset_data in data_pathcls.items():
#     for qa in subset_data:
#         if 'source_img' in qa.keys():
#             dst_path = '/data/dataset/vqa/PathMMU/images/' + qa['img']
#             src_path = pathcls_img_dir + qa['source_img']
#             if src_path.split('.')[-1] in ['tif', 'tiff']:
#                 convert_tiff_to_jpeg(src_path, dst_path)
#             else:
#                 shutil.copy(src_path, dst_path)

pathcls_img_dir = '/data/dataset/vqa/PathMMU/PathMMU_add/'
data = load_data('/data/dataset/vqa/PathMMU/data.json')
# data = load_data('/data/dataset/vqa/PathMMU/data_new.json')
# data_pathcls = data['Atlas']
# data_pathcls = data['PathCLS']

for source_name, source_data in data.items(): # Atlas
    for subset, subset_data in source_data.items(): # val
        cnt = 0
        for qa in subset_data:
            # image, question, answer
            assert 'question' in qa.keys() and 'answer' in qa.keys()
            assert 'img' in qa.keys() or 'image' in qa.keys()
            assert not ('img' in qa.keys() and 'image' in qa.keys())
            img_name = qa['img'] if 'img' in qa.keys() else qa['image']
            img_path = f"/data/dataset/vqa/PathMMU/images/{img_name}"
            if not os.path.exists(img_path):
                cnt += 1
        print(f'{source_name} {subset} 共{len(subset_data)}个, {cnt}个图片不存在')

# QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format. Just one letter (A, B, C, or D) with no explanation or additional text in <answer> </answer>."
# QUESTION_TEMPLATE2 = "{Question} Please output only the final answer option directly. Just one letter (A, B, C, or D) with no explanation or additional text."
