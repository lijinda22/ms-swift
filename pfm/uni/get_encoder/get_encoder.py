import os
import logging
from os.path import join as pjoin
import json
import timm
import torch
import torch.nn as nn
from torchvision import transforms

def get_eval_transforms_uni(img_resize: int = 224):
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


def get_encoder_uni():
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


# def get_eval_transforms_uni2(img_resize: int = 224):
#     """
#     Get the evaluation transforms for UNI2.
#     """
#     transform = transforms.Compose(
#         [
#             transforms.Resize(img_resize),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ]
#     )
#     return transform


def get_encoder_uni2():
    """
    Get the UNI2 encoder model.
    """
    local_dir = "/data/ckpt/uni2/"
    
    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224, 
        'patch_size': 14, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0, 
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
    }
    
    model = timm.create_model(
        pretrained=False, **timm_kwargs
    )
    
    checkpoint_path = pjoin(local_dir, "pytorch_model.bin")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    else:
        logging.warning(f"UNI2 checkpoint not found at {checkpoint_path}. Returning initialized model.")
        
    model.eval()
    return model