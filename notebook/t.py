# /data/ckpt/PathCap/processed_data.json
# 将这个json命名改为pathcap_pair_{len}.json, len是item的数量
# save: /data/ljd/VLM-R1/dataset/

import json
import os

# 输入文件路径
input_path = "/data/ckpt/PathCap/processed_data.json"

# 读取数据
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 获取数据长度
data_len = len(data)

# 构造新文件名
output_filename = f"pathcap_pair_{data_len}.json"

# 输出目录
output_dir = "/data/ljd/VLM-R1/dataset/"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 完整的输出路径
output_path = os.path.join(output_dir, output_filename)

# 保存数据
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"文件已保存到: {output_path}")
