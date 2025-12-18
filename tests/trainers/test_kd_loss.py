import torch
import torch.nn as nn
from swift.trainers.trainers import SftKdTrainer, KnowledgeDistillationModule
import shutil
import tempfile
import unittest

class MockVisual:
    def __init__(self, hidden_size):
        self.merger = type('Merger', (), {'linear_fc2': type('FC', (), {'out_features': hidden_size})()})()
        self.spatial_merge_size = 2

class MockModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.model = type('InnerModel', (), {'visual': MockVisual(hidden_size)})()
        self.config = type('Config', (), {})()
        self.training = True
        self.model_info = type('Info', (), {'is_moe_model': False})()
        self.generation_config = type('GC', (), {'max_new_tokens': 10})
        self.dtype = torch.float32
        
    def forward(self, **kwargs):
        # Fake output
        loss = torch.tensor(1.0, requires_grad=True)
        return type('Output', (), {'loss': loss, 'logits': None, 'aux_loss': None})()

class MockTeacher(nn.Module):
    def __init__(self, embed_dim, patch_size=14):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = type('PE', (), {'patch_size': patch_size})()
        # Mock parameters
        self.param = nn.Parameter(torch.randn(1, embed_dim))
        
    def forward_features(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        num_patches = (H // self.patch_embed.patch_size) * (W // self.patch_embed.patch_size)
        # Return CLS + Patches
        return torch.randn(B, 1 + num_patches, self.embed_dim).to(x.device)

class TestKD(unittest.TestCase):
    def test_kd_step(self):
        student_dim = 128
        teacher_dim = 256
        batch_size = 2
        
        model = MockModel(student_dim)
        teacher = MockTeacher(teacher_dim)
        
        # Mock Trainer arguments
        args = type('Args', (), {
            'output_dir': tempfile.mkdtemp(),
            'gradient_accumulation_steps': 1,
            'past_index': -1,
            'enable_dft_loss': False,
            'enable_channel_loss': False,
            'label_smoother': None,
            'average_tokens_across_devices': False,
            'dataloader_num_workers': 0,
            'tuner_backend': 'swift',
            'predict_with_generate': False,
            'per_device_eval_batch_size': 1,
            'device': 'cpu',
            'use_liger_kernel': False,
            'router_aux_loss_coef': 0.0,
            'resume_from_checkpoint': None
        })()
        
        sft_args = type('SftArgs', (), {
            'kd_loss_weight': 1.0,
            'kd_token_strategy': 'patch_mse',
            'kd_weight_strategy': 'similarity_weighted'
        })()
        
        trainer = SftKdTrainer(
            model=model,
            args=args,
            sft_args=sft_args,
            teacher_models=[teacher],
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None
        )
        
        # Manually trigger hook
        # Student tokens: (Total_Patches, D_s)
        # Assume 2 images, 14x14 patches each -> 196 patches.
        # But wait, logic depends on image_grid_thw
        
        # Let's say inputs have image_grid_thw
        # H,W = 28, 28 -> 2x2 grid of 14x14 patches? No.
        # Original logic: H, W of image.
        # patch_size=14.
        
        # Input Mock
        h, w = 28, 42
        inputs = {
            'labels': torch.zeros(batch_size, 10),
            'pixel_values': torch.randn(batch_size, 3, h, w),
            # teacher_pixel_values: List[List[Tensor]] (Outer: Teacher, Inner: Batch)
            'teacher_pixel_values': [[torch.randn(3, h, w) for _ in range(batch_size)]], 
            'teacher_image_indices': [torch.arange(batch_size)],     # list of tensors
            'image_grid_thw': torch.tensor([[1, h, w], [1, h, w]]) # (B, 3)
        }
        
        # Student Output Mock (Global Buffer)
        # Size = B * (H//merge_size) * (W//merge_size)
        # merge_size = 2 default
        # h=28, w=28 -> 14x14 grid -> 196 tokens per image
        total_tokens = batch_size * (h // 2) * (w // 2)
        trainer.student_cls_token_buffer = torch.randn(total_tokens, student_dim, requires_grad=True)
        
        # Run Compute Loss
        loss = trainer.compute_loss(model, inputs)
        print(f"Computed Loss: {loss}")
        
        loss.backward()
        print("Backward successful")
        
        # Check grads
        self.assertTrue(trainer.student_cls_token_buffer.grad is not None)
        kd_mod = trainer.kd_modules[0]
        self.assertTrue(kd_mod.student_projection.weight.grad is not None)
        self.assertTrue(kd_mod.teacher_query_proj.weight.grad is not None)
        self.assertTrue(kd_mod.teacher_key_proj.weight.grad is not None)
        
        shutil.rmtree(args.output_dir)

if __name__ == '__main__':
    unittest.main()
