import torch
import json
import os

def inspect_data():
    pt_path = '/data/ljd/Pathology_FM_LLM/expriment/classify/CCRCC/vlm_vision_only/train/feat/lingshu-7b.pt'
    json_path = '/data/ljd/Pathology_FM_LLM/expriment/classify/CCRCC/vlm_vision_only/train/label/lingshu-7b.json'

    print("=" * 50)
    print(f"正在加载 Tensor: {pt_path}")
    tensor_data = torch.load(pt_path, map_location='cpu')
    print(f"数据类型: {type(tensor_data)}")
    if isinstance(tensor_data, torch.Tensor):
        print(f"Tensor Shape: {tensor_data.shape}")
        print(f"Tensor Dtype: {tensor_data.dtype}")
    elif isinstance(tensor_data, dict):
        for k, v in tensor_data.items():
            if hasattr(v, 'shape'):
                print(f"  - Key: '{k}', Shape: {v.shape}")
            else:
                print(f"  - Key: '{k}', Value: {v}")
    else:
        print(f"内容: {tensor_data}")
        

    print("-" * 50)

    print(f"正在读取 JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    if isinstance(json_data, list):
        print(f"JSON 类型: List (列表)")
        print(f"列表长度: {len(json_data)}")
        if len(json_data) > 0:
            print("前 1 条数据示例:")
            print(json.dumps(json_data[0], indent=4, ensure_ascii=False))
    elif isinstance(json_data, dict):
        print(f"JSON 类型: Dict (字典)")
        print(f"Keys: {list(json_data.keys())}")
    else:
        print("JSON 内容预览:")
        print(json_data)
    print("=" * 50)

if __name__ == '__main__':
    inspect_data()