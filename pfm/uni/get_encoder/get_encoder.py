import os
import logging
from os.path import join as pjoin
import json
import timm
import torch
import torch.nn as nn
from torchvision import transforms
import sys
from torchvision.transforms.functional import to_pil_image
sys.path.append("../../../")

def get_eval_transforms_uni(img_resize: int = 224):
    eval_transform = []
    if (img_resize is not None) and img_resize > 0:
        eval_transform.append(transforms.Resize(img_resize))
        # eval_transform.append(transforms.CenterCrop(img_resize))
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

def get_encoder_uni2():
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
    checkpoint_path = "/data/ckpt/uni2/pytorch_model.bin"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    model.eval()
    return model


def get_eval_transforms_virchow2(img_resize: int = 224):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [
            transforms.Resize(img_resize, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(img_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transform


def get_encoder_virchow2():
    timm_kwargs = {
        'model_name': 'vit_huge_patch14_224',
        'img_size': 224, 
        'patch_size': 14, 
        'init_values': 1e-5, 
        'num_classes': 0, 
        'dynamic_img_size': True,
        'mlp_ratio': 5.3375,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 4, 
    }
    model = timm.create_model(
        pretrained=False, **timm_kwargs
    )
    checkpoint_path = "/data/ckpt/virchow2/pytorch_model.bin"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


if __name__ == "__main__":
    def check_model(name, model_fn, transform_fn):
        print(f"\n{'='*10} {name} {'='*10}")
        model = model_fn().cuda().eval()
        transform = transform_fn()
        
        # 构造输入
        img = transform(to_pil_image(torch.rand(3, 448, 224))).unsqueeze(0).cuda()
        with torch.no_grad():
            feat = model.forward_features(img)
            
        reg_tokens = getattr(model, 'reg_tokens', 0)
        prefix_tokens = getattr(model, 'num_prefix_tokens', 0)
        # 通常 prefix_tokens = cls + reg
        has_cls = prefix_tokens > reg_tokens
        
        print(f"Feature Shape: {feat.shape}")
        print(f"[Info] prefix_tokens: {prefix_tokens}, reg_tokens: {reg_tokens}, has_cls: {has_cls}")
        
        layout = []
        if has_cls: layout.append("CLS(0)")
        if reg_tokens > 0: 
            start = 1 if has_cls else 0
            layout.append(f"REG({start}~{start+reg_tokens-1})")
        
        patch_start = prefix_tokens
        layout.append(f"PATCH({patch_start}~END)")
        
        print(f"Token Layout: {' + '.join(layout)}")
        print("patch_size: ", model.patch_embed.patch_size)

    check_model("Virchow2", get_encoder_virchow2, get_eval_transforms_virchow2)
    check_model("UNI2", get_encoder_uni2, get_eval_transforms_uni)
