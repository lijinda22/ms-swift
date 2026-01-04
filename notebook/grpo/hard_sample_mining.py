import os
import json
import random
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from swift.llm import get_model_tokenizer, get_template, Template
from swift.utils import get_logger
import numpy as np
import matplotlib.pyplot as plt


logger = get_logger()


def selective_prob(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    针对输入的 token_ids，从 logits 中提取对应的概率。
    logits: (batch, seq_len, vocab_size)
    input_ids: (batch, seq_len)
    """
    probs = F.softmax(logits, dim=-1)
    return probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


class HardSampleMiner:
    def __init__(self, model_id_or_path: str, model_type: str = None, torch_dtype: str = 'bfloat16', device: str = 'auto'):
        """
        初始化挖掘器。
        Args:
            model_id_or_path: SFT 后的模型路径。
            model_type: 模型类型。
            torch_dtype: 推理精度。
            device: 设备配置，支持 'cuda', 'cpu', 'auto'。使用 'auto' 可开启多卡推理 (Model Parallelism)。
        """
        self.model, self.tokenizer = get_model_tokenizer(
            model_id_or_path,
            getattr(torch, torch_dtype),
            model_type=model_type,
            device_map=device
        )
        self.model.eval()
        # Fix: When device='auto', we cannot use the string 'auto' for tensor.to(). 
        # We must use the model's actual device.
        self.device = self.model.device
        logger.info(f"Model loaded on device: {self.device} (Request: {device})")
        
        self.template: Template = get_template(self.model.model_meta.template, self.tokenizer)
        self.template.set_mode('train')

    @torch.no_grad()
    def calculate_batch_ground_truth_avg_prob(self, batch_messages: List[List[Dict[str, str]]], batch_images: List[List[str]] = None) -> List[float]:
        """
        批量使用 Teacher Forcing 计算数据集中 Ground Truth 答案的平均概率。
        """
        # Ensure mode is train for label generation
        self.template.set_mode('train')
        
        pad_token_id = self.tokenizer.pad_token_id
        
        # 1. Encode all messages
        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        max_len = 0
        
        has_images = batch_images is not None and any(img for img in batch_images)

        for i, messages in enumerate(batch_messages):
            images = batch_images[i] if has_images else None
            # Prepare inputs for template.encode
            # Swift template usually expects 'images' key in kwargs or in the dict if inputs is a dict
            # We follow the standard pattern: template.encode({'messages': ..., 'images': ...})
            input_data = {'messages': messages}
            if images:
                input_data['images'] = images
            
            encoded = self.template.encode(input_data)
            input_ids = encoded['input_ids']
            labels = encoded['labels']
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            
            if 'pixel_values' in encoded:
                all_pixel_values.append(encoded['pixel_values'])
            if 'image_grid_thw' in encoded:
                all_image_grid_thw.append(encoded['image_grid_thw'])

            max_len = max(max_len, len(input_ids))
            
        # 2. Pad and stack (right padding for training/logprob calc usually fine, but ensure alignment)
        batch_input_ids = []
        batch_labels = []
        
        for input_ids, labels in zip(all_input_ids, all_labels):
            pad_len = max_len - len(input_ids)
            batch_input_ids.append(input_ids + [pad_token_id] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)
            
        input_ids_tensor = torch.tensor(batch_input_ids).to(self.device)
        labels_tensor = torch.tensor(batch_labels).to(self.device)
        
        # Prepare model kwargs
        model_kwargs = {}
        if all_pixel_values:
            # Concatenate pixel_values along appropriate dim. 
            # Reviewing typical Qwen-VL usage in Swift: 
            # They are usually concatenated to a single tensor (N_total_images, C, H, W) or similar.
            # encoded['pixel_values'] is typically a tensor.
            try:
                model_kwargs['pixel_values'] = torch.cat(all_pixel_values, dim=0).to(self.device)
            except Exception as e:
                logger.error(f"Error concatenating pixel_values: {e}")
        
        if all_image_grid_thw:
            try:
                model_kwargs['image_grid_thw'] = torch.cat(all_image_grid_thw, dim=0).to(self.device)
            except Exception as e:
                logger.error(f"Error concatenating image_grid_thw: {e}")

        # 3. Model Forward
        outputs = self.model(input_ids=input_ids_tensor, **model_kwargs)
        logits = outputs.logits
        
        # 4. Shift and Calculate Probs
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels_tensor[:, 1:].contiguous()
        shift_input_ids = input_ids_tensor[:, 1:].contiguous()
        
        probs = selective_prob(shift_logits, shift_input_ids)
        
        # 5. Calculate Average Prob per Sample
        batch_avg_probs = []
        for i in range(probs.size(0)):
            mask = (shift_labels[i] != -100)
            target_probs = probs[i][mask]
            
            if target_probs.numel() == 0:
                batch_avg_probs.append(0.0) # Or some other default value
            else:
                batch_avg_probs.append(target_probs.mean().item())
                
        return batch_avg_probs

    @torch.no_grad()
    def batch_generate(self, batch_messages: List[List[Dict[str, str]]], batch_images: List[List[str]] = None) -> List[str]:
        """
        批量生成回复。
        """
        # Switch to inference mode for generation (no labels needed)
        self.template.set_mode('pt')
        
        pad_token_id = self.tokenizer.pad_token_id
        
        # 1. Prepare prompts (remove the last assistant message if present, or just use as is if it's prompt-only)
        # Assuming input batch_messages are full conversation history [User, Assistant], we need [User].
        # Or if the caller expects us to handle it. 
        # Here we follow standard practice: prompt is everything BEFORE the response we want to check.
        # But wait, hard sample mining usually looks at correct answer vs pred.
        # The input `batch_messages` in `process_dataset` contains the FULL history including GT answer.
        # We must slice it.
        
        batch_prompts = []
        for msgs in batch_messages:
            # Slice to exclude the last assistant message (the GT answer)
            if msgs[-1]['role'] == 'assistant':
                batch_prompts.append(msgs[:-1])
            else:
                batch_prompts.append(msgs)

        # 2. Encode prompts
        all_input_ids = []
        all_pixel_values = []
        all_image_grid_thw = []
        max_len = 0
        
        has_images = batch_images is not None and any(img for img in batch_images)

        for i, msgs in enumerate(batch_prompts):
            images = batch_images[i] if has_images else None
            input_data = {'messages': msgs}
            if images:
                input_data['images'] = images
                
            encoded = self.template.encode(input_data)
            all_input_ids.append(encoded['input_ids'])
            
            if 'pixel_values' in encoded:
                all_pixel_values.append(encoded['pixel_values'])
            if 'image_grid_thw' in encoded:
                all_image_grid_thw.append(encoded['image_grid_thw'])
                
            max_len = max(max_len, len(encoded['input_ids']))
            
        # 3. Left Pad for generation (simplest for HF generate)
        batch_input_ids = []
        attention_masks = []
        
        for input_ids in all_input_ids:
            pad_len = max_len - len(input_ids)
            # Left padding
            padded_ids = [pad_token_id] * pad_len + input_ids
            mask = [0] * pad_len + [1] * len(input_ids)
            batch_input_ids.append(padded_ids)
            attention_masks.append(mask)
            
        input_ids_tensor = torch.tensor(batch_input_ids).to(self.device)
        attention_mask_tensor = torch.tensor(attention_masks).to(self.device)
        
        # Prepare model kwargs
        model_kwargs = {}
        if all_pixel_values:
            try:
                model_kwargs['pixel_values'] = torch.cat(all_pixel_values, dim=0).to(self.device)
            except Exception as e:
                logger.error(f"Error concatenating pixel_values in generate: {e}")
        
        if all_image_grid_thw:
            try:
                model_kwargs['image_grid_thw'] = torch.cat(all_image_grid_thw, dim=0).to(self.device)
            except Exception as e:
                logger.error(f"Error concatenating image_grid_thw in generate: {e}")

        # 4. Generate
        generated_ids = self.model.generate(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            max_new_tokens=64, # Optimized: Reduced from 256. PathVQA answers are usually short.
            pad_token_id=pad_token_id,
            do_sample=False,   # Optimized: Greedy decoding is faster and deterministic
            num_beams=1,
            use_cache=True,
            **model_kwargs
        )
        
        # 5. Decode (only new tokens)
        # generated_ids contains [input_ids + new_tokens]. We slice.
        new_tokens = generated_ids[:, max_len:]
        preds = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        return preds

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        return text.strip().lower().replace('.', '').replace(',', '')

    def process_dataset(self, 
                        input_path: str, 
                        output_path: str, 
                        threshold: float = 0.5, # Definition of "High Confidence"
                        keep_ratio: float = None, # Deprecated/Secondary if we use threshold+correctness
                        batch_size: int = 8,
                        limit: int = None):
        """
        处理数据集并过滤简单样本。
        简单样本定义: (Avg Prob > threshold) AND (Pred == GT)
        """
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        logger.info(f"Loaded {len(data)} samples from {input_path}")
        
        if limit and len(data) > limit:
            logger.info(f"Randomly sampling {limit} items from {len(data)} total.")
            data = random.sample(data, limit)
        
        # Optimization: Sort data by length to minimize padding overhead
        # We estimate length by string dump size or just message content length
        logger.info("Sorting data by length to optimize batch processing...")
        data.sort(key=lambda x: len(json.dumps(x['messages'])))
        
        results = []
        
        # Batch processing
        for i in tqdm(range(0, len(data), batch_size), desc="Processing Batches"):
            batch_data = data[i:i + batch_size]
            batch_messages = []
            batch_images = []
            valid_indices = []
            batch_gt_answers = []
            
            for j, item in enumerate(batch_data):
                messages = item.get('messages', [])
                images = item.get('images', []) # Extract images
                
                if messages and messages[-1]['role'] == 'assistant':
                    batch_messages.append(messages)
                    batch_images.append(images)
                    valid_indices.append(j)
                    batch_gt_answers.append(messages[-1]['content'])
            
            if not batch_messages:
                continue
            
            # 1. Calculate Probs (Teacher Forcing)
            batch_avg_probs = self.calculate_batch_ground_truth_avg_prob(batch_messages, batch_images)
            
            # 2. Generate Predictions
            batch_preds = self.batch_generate(batch_messages, batch_images)
            
            for idx, avg_prob, pred_text, gt_text in zip(valid_indices, batch_avg_probs, batch_preds, batch_gt_answers):
                item = batch_data[idx]
                item['avg_prob'] = avg_prob
                item['pred_text'] = pred_text
                item['is_correct'] = self._normalize_text(pred_text) == self._normalize_text(gt_text)
                item['is_easy'] = (avg_prob > threshold) and item['is_correct']
                
                if not item['is_easy']:
                    results.append(item)

        # Analysis & Visualization
        full_data = [d for d in data if 'avg_prob' in d] # Only processed ones
        if not full_data: return

        probs = [d['avg_prob'] for d in full_data]
        corrects = [d['is_correct'] for d in full_data]
        
        # 1. Prob Distribution
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist([p for p, c in zip(probs, corrects) if c], bins=50, alpha=0.5, label='Correct', color='green')
        plt.hist([p for p, c in zip(probs, corrects) if not c], bins=50, alpha=0.5, label='Incorrect', color='red')
        plt.title('Prob Distribution: Correct vs Incorrect')
        plt.xlabel('Avg Prob')
        plt.legend()
        
        # 2. Correctness Ratio
        plt.subplot(1, 2, 2)
        counts = [sum(corrects), len(corrects) - sum(corrects)]
        plt.pie(counts, labels=['Correct', 'Incorrect'], autopct='%1.1f%%', colors=['green', 'red'])
        plt.title(f'Correctness Ratio (Total: {len(full_data)})')
        
        plot_path = os.path.join(os.path.dirname(output_path), 'analysis_plot.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved analysis plot to {plot_path}")

        # Save Files
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 1. Filtered Hard Samples
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Saved filtered dataset ({len(results)}) to {output_path}")

        # 2. Full Analysis Data
        full_path = output_path.replace('.jsonl', '_full_analysis.jsonl')
        with open(full_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Saved full analysis dataset to {full_path}")


if __name__ == "__main__":
    model_path = '/data/ckpt/Qwen3-VL-2B-Instruct'
    input_dataset = '/data/ljd/VLM-R1/dataset/sft/swiftsft_dataset_new/pathvqa_eval_3016.jsonl'
    output_dir = '/data/ljd/VLM-R1/dataset/rl/hard/'
    output_file = os.path.join(output_dir, 'pathvqa_eval_3016_hard.jsonl')

    miner = HardSampleMiner(model_id_or_path=model_path)
    miner.process_dataset(
        input_path=input_dataset,
        output_path=output_file,
        threshold=0.6,
        batch_size=8,  # Recommended: Increased batch size for speed
        limit=64     # Optional: Sample k items (e.g., 1000) for fast testing
    )
    print(f"Done! Filtered dataset saved to: {output_file}")