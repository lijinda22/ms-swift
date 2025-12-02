import json

from urllib.request import urlopen
from PIL import Image
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

LOCAL_DIR = "/data/ckpt/BiomedCLIP"
model_name = "biomedclip_local"
with open(f"{LOCAL_DIR}/open_clip_config.json", "r") as f:
    config = json.load(f)
    model_cfg = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]
if (
    not model_name.startswith(HF_HUB_PREFIX)
    and model_name not in _MODEL_CONFIGS
    and config is not None
):
    _MODEL_CONFIGS[model_name] = model_cfg

model, _, preprocess = create_model_and_transforms(
    model_name=model_name,
    pretrained=f"{LOCAL_DIR}/open_clip_pytorch_model.bin",
    **{f"image_{k}": v for k, v in preprocess_cfg.items()},
)
model = model.cuda().eval()

random_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
processed_image = preprocess(Image.fromarray(random_array)).unsqueeze(0).cuda()
with torch.no_grad():
    image_features = model.encode_image(processed_image)
    normalized_features = image_features / image_features.norm(dim=1, keepdim=True)

features_np = normalized_features.cpu().numpy()
print(f"特征维度: {features_np.shape}")
print(f"特征前5值: {features_np[0, :5]}")
