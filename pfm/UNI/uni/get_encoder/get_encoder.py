import os
import logging
from os.path import join as pjoin
import json
import timm
import torch
import torch.nn as nn
from torchvision import transforms

def get_eval_transforms(img_resize: int = 224):
    eval_transform = []
    if (img_resize is not None) and img_resize > 0:
        eval_transform.append(transforms.Resize(img_resize))
        eval_transform.append(transforms.CenterCrop(img_resize))
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    eval_transform.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    eval_transform = transforms.Compose(eval_transform)
    return eval_transform


def get_encoder():
    ckpt_dirpath = "/data/ckpt/uni"
    cfg_json_path = pjoin(ckpt_dirpath, "config.json")
    local_tile_encoder_path = pjoin(ckpt_dirpath, "pytorch_model.bin")
    with open(cfg_json_path, "r") as f:
        cfg = json.load(f)
    cfg.pop("num_features", None)
    kwargs = {"model_name": cfg.pop("architecture"), **cfg.pop("model_args", {})}
    kwargs.update(cfg)
    print("kwargs: ", kwargs)
    tile_encoder = timm.create_model(**kwargs)
    state_dict = torch.load(local_tile_encoder_path, map_location="cpu")
    missing_keys, unexpected_keys = tile_encoder.load_state_dict(
        state_dict, strict=True
    )
    print("missing_keys:", missing_keys)
    print("unexpected_keys:", unexpected_keys)
    return tile_encoder