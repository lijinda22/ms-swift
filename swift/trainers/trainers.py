# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import os
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import EvalPrediction
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available
import torch.nn.functional as F
from conchv1_5 import create_model_from_pretrained
from PIL import Image
from swift.utils import (
    JsonlWriter,
    Serializer,
    gc_collect,
    get_logger,
    unwrap_model_for_generation,
)
from .arguments import Seq2SeqTrainingArguments, TrainingArguments
from .mixin import DataLoaderMixin, SwiftMixin
from .utils import per_token_loss_func, per_token_loss_func_sp

logger = get_logger()


# 基础训练器类，继承自HuggingFace Trainer
class Trainer(SwiftMixin, DataLoaderMixin, HfTrainer):
    args: TrainingArguments

    @contextmanager
    def _patch_loss_function(self):
        # 修补损失函数以适配设备映射
        model = self.model
        if isinstance(model, PeftModel):
            model = model.model
        model_cls = model.__class__
        if not hasattr(model_cls, "loss_function"):
            yield
            return

        loss_function = model.loss_function
        _old_loss_function = model_cls.loss_function

        @staticmethod
        @wraps(loss_function)
        def new_loss_function(logits, labels, **kwargs):
            labels = labels.to(logits.device)  # fix device_map
            return loss_function(logits=logits, labels=labels, **kwargs)

        model_cls.loss_function = new_loss_function
        try:
            yield
        finally:
            model_cls.loss_function = _old_loss_function

    def train(self, *args, **kwargs):
        # 训练模型主函数
        with self._patch_loss_function():
            return super().train(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # 计算模型损失值
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if inputs.get("labels") is not None:
            self._compute_acc(outputs, inputs["labels"])
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss


def gather_for_unpadded_tensors(input_data, use_gather_object=False):
    # 收集未填充的张量数据用于评估
    from accelerate.utils import gather_object

    input_data = gather_object(input_data)
    output = []
    for _data in input_data:
        if len(_data.shape) == 0:
            _data = _data.unsqueeze(0)
        _data = _data.cpu()
        output.append(_data)
    if len(output[0].shape) == 1 and output[0].shape[0] > 1:
        data = torch.stack(output, dim=0)
    else:
        data = torch.concat(output, dim=0)
    return data


# 嵌入模型专用训练器
class EmbeddingTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.calculate_metric
        self.preprocess_logits_for_metrics = None
        self.label_names = ["labels"]
        self.gather_function = gather_for_unpadded_tensors

    def evaluation_loop(self, *args, **kwargs):
        # 评估循环
        output = super().evaluation_loop(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        return output

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        # 计算嵌入模型评估指标
        from swift.plugin.loss import (
            calculate_paired_metrics,
            calculate_infonce_metrics,
        )

        args = self.args
        if args.loss_type == "infonce":
            return calculate_infonce_metrics(
                eval_prediction.predictions, eval_prediction.label_ids
            )
        else:
            return calculate_paired_metrics(
                eval_prediction.predictions, eval_prediction.label_ids
            )


# 重排序模型专用训练器
class RerankerTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.calculate_metric
        self.label_names = ["labels"]

        # 为生成式重排序器设置日志预处理以减少内存使用
        if self.args.loss_type in {
            "generative_reranker",
            "listwise_generative_reranker",
        }:
            self.preprocess_logits_for_metrics = (
                self._preprocess_generative_reranker_logits
            )
        else:
            self.preprocess_logits_for_metrics = None
        self.gather_function = gather_for_unpadded_tensors

    def _preprocess_generative_reranker_logits(self, logits, labels):
        # 预处理生成式重排序器的日志，仅提取必要部分以节省内存
        import torch
        import os

        # 获取正负标记的token ID
        positive_token = os.environ.get("GENERATIVE_RERANKER_POSITIVE_TOKEN", "yes")
        negative_token = os.environ.get("GENERATIVE_RERANKER_NEGATIVE_TOKEN", "no")

        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            # 回退：如果无法获取tokenizer则返回完整logits
            return logits

        try:
            positive_token_id = tokenizer.convert_tokens_to_ids(positive_token)
            negative_token_id = tokenizer.convert_tokens_to_ids(negative_token)
        except Exception:
            # 回退：如果token转换失败则返回完整logits
            return logits

        # 从每个样本的最后一个有效位置提取正/负token的日志
        # 形状: logits [batch, seq_len, vocab]
        if len(logits.shape) == 3:
            batch_size, _, vocab_size = logits.shape

            # 识别填充行（整个词汇表日志都是-100）
            row_is_pad = (logits == -100).all(dim=-1)  # [batch, seq_len]
            valid_mask = ~row_is_pad
            lengths = valid_mask.long().sum(dim=1) - 1
            lengths = torch.clamp(lengths, min=0)
            last_indices = lengths.to(device=logits.device)

            # 收集每个样本最后一个有效索引处的日志: [batch, vocab]
            gather_index = last_indices.view(batch_size, 1, 1).expand(
                batch_size, 1, vocab_size
            )
            last_step_logits = torch.gather(logits, dim=1, index=gather_index).squeeze(
                1
            )

            positive_logits = last_step_logits[:, positive_token_id]
            negative_logits = last_step_logits[:, negative_token_id]
            logits = positive_logits - negative_logits
            return logits
        else:
            # 意外形状，按原样返回
            return logits

    def evaluation_loop(self, *args, **kwargs):
        # 评估循环
        output = super().evaluation_loop(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        return output

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        # 计算重排序模型评估指标
        from swift.plugin.loss import calculate_reranker_metrics

        return calculate_reranker_metrics(
            eval_prediction.predictions, eval_prediction.label_ids
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # 计算重排序模型损失值
        # 检查是否有自定义损失函数
        if self.compute_loss_func is not None:
            # 获取标签并计算输出
            labels = inputs.get("labels")
            if labels is not None:
                labels = inputs.pop("labels")

            outputs = model(**inputs)

            if labels is not None:
                # 调用自定义损失函数
                loss = self.compute_loss_func(
                    outputs, labels, num_items_in_batch=num_items_in_batch, trainer=self
                )
            else:
                # 回退到模型的损失计算
                loss = outputs.loss

            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                loss = loss / self.args.gradient_accumulation_steps

            if labels is not None:
                self._compute_acc(outputs, labels)

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )


# 序列到序列模型训练器
class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    args: Seq2SeqTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        if self.args.predict_with_generate:
            from swift.llm import PtEngine

            self.infer_engine = PtEngine.from_model_template(
                self.model,
                self.template,
                max_batch_size=self.args.per_device_eval_batch_size,
            )
        self.jsonl_writer = JsonlWriter(
            os.path.join(self.args.output_dir, "predict.jsonl")
        )

    @staticmethod
    def _predict_data_collator(batch):
        # 预测数据整理器
        return {"_data": batch}

    @contextmanager
    def _patch_predict_with_generate(self):
        # 修补预测时的生成过程
        origin_data_collator = self.data_collator
        self.data_collator = self._predict_data_collator
        packing = self.template.packing
        padding_free = self.template.padding_free
        self.template.packing = False
        self.template.padding_free = False
        try:
            yield
        finally:
            self.template.packing = packing
            self.template.padding_free = padding_free
            self.data_collator = origin_data_collator

    def evaluate(self, *args, **kwargs):
        # 模型评估
        context = (
            self._patch_predict_with_generate()
            if self.args.predict_with_generate
            else nullcontext()
        )
        with context:
            res = super().evaluate(*args, **kwargs)
            gc_collect()
            return res

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 预测步骤
        if not self.args.predict_with_generate or prediction_loss_only:
            with self.template.forward_context(self.model, inputs):
                return super().prediction_step(
                    model,
                    inputs,
                    prediction_loss_only=prediction_loss_only,
                    ignore_keys=ignore_keys,
                )
        from swift.llm import RequestConfig, InferRequest

        data_list = inputs["_data"]
        labels_list = [
            InferRequest.remove_response(data["messages"]) for data in data_list
        ]
        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ), self.template.generate_context():
            resp_list = self.infer_engine.infer(
                data_list,
                RequestConfig(max_tokens=self.model.generation_config.max_new_tokens),
                use_tqdm=False,
                template=self.template,
            )

        response_list = []
        jsonl_cache = []
        device = self.args.device
        for data, resp, labels in zip(data_list, resp_list, labels_list):
            response = resp.choices[0].message.content
            jsonl_cache.append({"response": response, "labels": labels, **data})
            response_list.append(
                Serializer.to_tensor(resp.choices[0].message.content).to(device=device)
            )
        self.jsonl_writer.append(jsonl_cache, gather_obj=True)
        labels_list = [
            Serializer.to_tensor(labels).to(device=device) for labels in labels_list
        ]
        response_list = pad_sequence(response_list, batch_first=True, padding_value=0)
        labels_list = pad_sequence(labels_list, batch_first=True, padding_value=0)
        return None, response_list, labels_list

    def _prepare_inputs(self, inputs):
        # 准备输入数据
        from swift.llm import HfConfigFactory

        args = self.args
        inputs = super()._prepare_inputs(inputs)
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel

            sequence_parallel.prepare_inputs(inputs)

        use_logits_to_keep = self.get_use_logits_to_keep(
            self.template.sequence_parallel_size == 1
        )
        if use_logits_to_keep:
            self.prepare_logits_to_keep(inputs)
            if args.tuner_backend == "unsloth" and isinstance(
                inputs["logits_to_keep"], torch.Tensor
            ):
                inputs["logits_to_keep"] = int(inputs["logits_to_keep"].sum())

        base_model = self.template.get_base_model(self.model)
        if (
            self.model.model_info.is_moe_model
            and "output_router_logits"
            in inspect.signature(base_model.forward).parameters
        ):
            HfConfigFactory.set_config_attr(
                base_model.config, "router_aux_loss_coef", args.router_aux_loss_coef
            )
            base_model.router_aux_loss_coef = args.router_aux_loss_coef
            logger.info_once(f"router_aux_loss_coef: {args.router_aux_loss_coef}")
            if args.router_aux_loss_coef > 0:
                inputs["output_router_logits"] = True
        inputs["compute_loss_func"] = self.compute_loss_func
        return inputs

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # 计算序列到序列模型损失值
        labels = None
        compute_loss_func: Callable = inputs.pop("compute_loss_func", None)
        loss_scale = inputs.pop("loss_scale", None)
        text_position_ids = inputs.pop("text_position_ids", None)
        if text_position_ids is None:
            text_position_ids = inputs.get("position_ids")
        channels = inputs.pop("channel", None)

        if (
            self.label_smoother is not None
            or compute_loss_func is not None
            or loss_scale is not None
            or self.args.enable_dft_loss
            or self.args.enable_channel_loss
            or self.template.sequence_parallel_size > 1
        ) and "labels" in inputs:
            if self.args.use_liger_kernel:
                logger.warning_once(
                    "The cross_entropy loss function defined in Liger Kernel will not "
                    "take effect, potentially leading to increased GPU memory consumption."
                )
            labels = inputs.pop("labels")
        outputs = model(**inputs)
        if getattr(outputs, "aux_loss", None) is not None:
            mode = "train" if self.model.training else "eval"
            self.custom_metrics[mode]["aux_loss"].update(outputs.aux_loss)
        # 保存过去状态（如果存在）
        # TODO: 这需要修复并稍后清理
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs["labels"]
            outputs.loss = outputs.loss.to(labels.device)
            # 修复 https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * (
                    (labels[:, 1:] != -100).sum() / num_items_in_batch
                )

            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # 我们不在此处使用.loss，因为模型可能返回元组而不是ModelOutput
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        else:
            outputs.loss = None
            if (
                self.args.enable_dft_loss
                or loss_scale is not None
                or self.args.enable_channel_loss
                or self.template.sequence_parallel_size > 1
            ):
                if self.template.sequence_parallel_size > 1:
                    outputs.loss = per_token_loss_func_sp(
                        outputs, labels, enable_dft_loss=self.args.enable_dft_loss
                    )
                else:
                    outputs.loss = per_token_loss_func(
                        outputs, labels, enable_dft_loss=self.args.enable_dft_loss
                    )

                if loss_scale is not None:
                    loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)
                    outputs.loss = outputs.loss * loss_scale

                if self.args.enable_channel_loss and channels is not None:
                    mode = "train" if self.model.training else "eval"
                    metrics = self.custom_metrics[mode]
                    masks = torch.roll(labels, shifts=-1, dims=-1).view(-1) != -100
                    if self.template.padding_free:
                        cu_seqlens = self.get_cu_seqlens(
                            text_position_ids, inputs.get("logits_to_keep")
                        )
                    else:
                        cu_seqlens = (
                            torch.arange(0, labels.shape[0] + 1) * labels.shape[1]
                        )
                    for i in range(cu_seqlens.shape[0] - 1):
                        channel = channels[i]
                        slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                        metrics[f"loss_{channel}"].update(
                            outputs.loss[slice_][masks[slice_]]
                        )

            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # 用户自定义的compute_loss函数
            if compute_loss_func is not None:
                loss = compute_loss_func(
                    outputs, labels, num_items_in_batch=num_items_in_batch, trainer=self
                )
            elif self.label_smoother is None:
                # 处理由loss_scale生成的outputs.loss
                if num_items_in_batch is None:
                    num_items_in_batch = (labels[:, 1:] != -100).sum()
                loss = outputs.loss.sum() / num_items_in_batch
            else:
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)

            if (
                self.model.model_info.is_moe_model
                and self.args.router_aux_loss_coef is not None
            ):
                aux_loss = outputs.get("aux_loss")
                if aux_loss is not None:
                    if num_items_in_batch is not None:
                        aux_loss = aux_loss * (
                            (labels[:, 1:] != -100).sum() / num_items_in_batch
                        )
                    loss = loss + self.args.router_aux_loss_coef * aux_loss.to(
                        loss.device
                    )

        if (
            getattr(self.args, "average_tokens_across_devices", False)
            and self.model_accepts_loss_kwargs
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        if (
            outputs.logits is not None
            and labels is not None
            and self.args.tuner_backend != "unsloth"
        ):
            # Liger没有logits
            # Unsloth在输出logits方面有bug
            self._compute_acc(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, *args, **kwargs):
        # 训练步骤
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)


# SFT + ViT知识蒸馏训练器
class KnowledgeDistillationModule(nn.Module):
    """A module to hold all components for Knowledge Distillation."""

    def __init__(
        self, student_hidden_size: int, teacher_hidden_size: int
    ):
        super().__init__()
        # Optimal Projection Strategy:
        # Teacher: Identity (preserve original semantic space)
        # Student: Learn to map to Teacher's space
        self.student_projection = nn.Linear(student_hidden_size, teacher_hidden_size)
        self.teacher_projection = nn.Identity()

        # Explicitly initialize student projection
        torch.nn.init.orthogonal_(self.student_projection.weight)
        if self.student_projection.bias is not None:
            torch.nn.init.zeros_(self.student_projection.bias)

        self.register_buffer("center", torch.zeros(1, teacher_hidden_size))


class SftKdTrainer(Seq2SeqTrainer):
    """
    Trainer for SFT + ViT Knowledge Distillation.
    Inherits from Seq2SeqTrainer.
    """

    def __init__(
        self,
        *args,
        sft_args: "TrainArguments",
        teacher_model: nn.Module,
        teacher_transform: Callable,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sft_args = sft_args
        self.teacher_model = teacher_model.eval()
        self.teacher_transform = teacher_transform
        self.kd_loss_weight = sft_args.kd_loss_weight
        self.student_T = sft_args.kd_student_T
        self.teacher_T = sft_args.kd_teacher_T
        self.center_momentum = sft_args.kd_center_momentum
        self.student_cls_token_buffer = None
        self.student_spatial_merge_size = 2  # Default for Qwen2-VL/3-VL
        self.kd_module: Optional[KnowledgeDistillationModule] = None
        self.original_data_collator = self.data_collator
        self.data_collator = self._kd_data_collator
        self._init_kd_components()

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override save_model to also save the KD adapter weights.
        """
        super().save_model(output_dir, _internal_call)
        # Only save on the main process to avoid race conditions and errors on other ranks
        if self.is_world_process_zero() and output_dir and self.kd_module:
            # Ensure the directory exists before saving
            os.makedirs(output_dir, exist_ok=True)
            kd_path = os.path.join(output_dir, "kd_module.pt")
            torch.save(self.kd_module.state_dict(), kd_path)
            logger.info(f"Saved KD module to {kd_path}")


    def _kd_data_collator(self, batch: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        KD Data Collator:
        仅负责从 batch 中提取预处理好的 'teacher_pixel_values' 并堆叠。
        """
        # 1. 调用原始 collator 处理 Student 的输入
        # 注意：如果开启 packing，这里的 batch 已经是 packed 过的（但我们的自定义字段可能被丢弃）
        student_inputs = self.original_data_collator(batch, **kwargs)

        # 2. 提取 Teacher 输入
        # Handle packing (list of lists)
        if batch and isinstance(batch[0], list):
            flat_batch = [item for sublist in batch for item in sublist]
        else:
            flat_batch = batch

        teacher_vals = []
        teacher_indices = []
        
        # Track student image index (dense) to align with student_cls_token_buffer
        student_img_idx = 0
        
        # 遍历 batch
        for i, item in enumerate(flat_batch):
            # print("item.keys:", item.keys())
            # Determine if this sample contributes an image to the student model
            has_student_image = False
            num_images_in_sample = 0
            
            if 'pixel_values' in item and item['pixel_values'] is not None:
                has_student_image = True
                # Estimate number of images. For standard swift/qwen2-vl, it's usually 1 per sample or flattened.
                # If pixel_values is list, len is num images. If tensor 4D, shape[0].
                # For simplicity and common cases (1 image), we assume 1 unless list.
                pv = item['pixel_values']
                if isinstance(pv, list):
                    num_images_in_sample = len(pv)
                elif isinstance(pv, torch.Tensor) and pv.ndim == 4:
                    num_images_in_sample = pv.shape[0]
                else:
                    num_images_in_sample = 1

            if has_student_image:
                if 'teacher_pixel_values' in item:
                    teacher_vals.append(item['teacher_pixel_values'])
                    # Map to the FIRST image of this sample in the student buffer
                    teacher_indices.append(student_img_idx)
                else:
                     # Student has image but Teacher failed/missing.
                     # Log warning only once per run/file?
                     pass

                # Check for multi-image alignment risk
                if num_images_in_sample > 1:
                     logger.warning_once(
                        f"Sample {i} contains {num_images_in_sample} images. "
                        "Current KD logic aligns Teacher to the FIRST image of the sample."
                    )
                
                student_img_idx += num_images_in_sample

            elif "teacher_pixel_values" in item:
                logger.warning_once(f"Sample {i} has teacher_pixel_values but no pixel_values. Ignoring teacher image.")

        if teacher_vals:
            # Stack 成一个 Tensor [Batch_Size, C, H, W]
            student_inputs["teacher_pixel_values"] = torch.stack(teacher_vals)
            # 记录哪些样本有 Teacher 图片 (Indices into the dense student buffer)
            student_inputs["teacher_image_indices"] = torch.tensor(teacher_indices, dtype=torch.long)
        
        return student_inputs

    def _student_hook(self, module, input, output):
        """
        Forward hook to capture the student ViT's global representation.
        Just capture raw output. Post-processing moves to compute_loss.
        """
        if isinstance(output, tuple):
            self.student_cls_token_buffer = output[0]
        else:
            self.student_cls_token_buffer = output

    def _init_kd_components(self):
        # 初始化知识蒸馏组件
        try:
            # For Qwen3-VL, the visual module is at model.model.visual
            student_vit_module = self.model.model.visual
            # The output dimension of the visual part is the input to the LM.
            # We can get this from the visual merger's output features.
            student_hidden_size = student_vit_module.merger.linear_fc2.out_features
            # Capture spatial merge size for pooling calculation
            self.student_spatial_merge_size = getattr(student_vit_module, "spatial_merge_size", 2)
            
            student_vit_module.register_forward_hook(self._student_hook)
            logger.info(
                f"Registered forward hook on student ViT module: {student_vit_module.__class__.__name__}"
            )
        except AttributeError as e:
            logger.error(
                f"Failed to find student ViT (model.model.visual) or its components. KD is disabled. Error: {e}"
            )
            self.kd_loss_weight = 0.0
            return
        try:
            teacher_hidden_size = self.teacher_model.trunk.embed_dim
        except AttributeError as e:
            logger.error(
                f"Failed to get teacher ViT hidden size (teacher_model.trunk.embed_dim). KD is disabled. Error: {e}"
            )
            self.kd_loss_weight = 0.0
            return

        self.kd_module = KnowledgeDistillationModule(
            student_hidden_size=student_hidden_size,
            teacher_hidden_size=teacher_hidden_size,
        ).to(self.model.device)

        # CRITICAL: Attach to model so optimizer picks up the parameters!
        # We use a distinct name to avoid conflict with existing modules.
        if not hasattr(self.model, "kd_adapter"):
            self.model.kd_adapter = self.kd_module
        else:
            logger.warning("Model already has 'kd_adapter'. Using existing one (resume?).")
            self.kd_module = self.model.kd_adapter

        logger.info("Initialized Knowledge Distillation Components (Optimal Projection).")
        logger.info(f"  Student ViT Dim: {student_hidden_size} -> {teacher_hidden_size}")
        logger.info(
            f"  Teacher ViT Dim: {teacher_hidden_size} -> {teacher_hidden_size} (Identity)"
        )

    def _dino_loss(self, student_output, teacher_output):
        # DINO损失计算
        student_out = F.softmax(student_output / self.student_T, dim=-1)
        teacher_out = (teacher_output - self.kd_module.center) / self.teacher_T
        teacher_out = F.softmax(teacher_out, dim=-1).detach()
        kd_loss = -(teacher_out * torch.log(student_out + 1e-9)).sum(dim=-1).mean()
        if self.model.training:
            with torch.no_grad():
                batch_center = teacher_output.mean(dim=0, keepdim=True)
                self.kd_module.center = (
                    self.kd_module.center * self.center_momentum
                    + batch_center * (1 - self.center_momentum)
                )
        return kd_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # 计算带知识蒸馏的损失值
        teacher_pixel_values = inputs.pop("teacher_pixel_values", None)
        teacher_image_indices = inputs.pop("teacher_image_indices", None)
        
        # Capture image_grid_thw for split calculation before super() consumes/modifies inputs
        # Note: Qwen2/3-VL expects 'image_grid_thw'.
        image_grid_thw = inputs.get("image_grid_thw", None)

        self.student_cls_token_buffer = None
        sft_loss_outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        sft_loss = sft_loss_outputs[0]
        
        if (
            self.kd_loss_weight == 0.0
            or self.kd_module is None
            or teacher_pixel_values is None
            or self.student_cls_token_buffer is None
        ):
            if self.kd_loss_weight > 0.0 and teacher_pixel_values is None:
                logger.warning_once(
                    "No valid teacher images in batch. Skipping KD loss."
                )
            if self.kd_loss_weight > 0.0 and self.student_cls_token_buffer is None:
                logger.warning_once(
                    "Student ViT hook did not run (no images in batch?). Skipping KD loss."
                )
            return sft_loss_outputs if return_outputs else sft_loss
        
        # --- Student Feature Processing (Split & Pool) ---
        # Qwen3-VL/Qwen2.5-VL output is flattened. We need grid_thw to split.
        if image_grid_thw is None:
             logger.warning_once("KD Error: image_grid_thw missing in inputs. Cannot split flattened student features. Skipping KD.")
             return sft_loss_outputs if return_outputs else sft_loss

        hidden_states = self.student_cls_token_buffer
        
        # Calculate split sizes based on merger logic
        # split_sizes = grid_thw.prod(-1) // (spatial_merge_size**2)
        # image_grid_thw shape: (Num_Images, 3) -> [T, H, W]
        # We need to ensure image_grid_thw is on the same device for calculation or move to cpu
        
        try:
            split_sizes = (image_grid_thw.to(hidden_states.device).prod(dim=-1) // (self.student_spatial_merge_size ** 2)).tolist()
            
            if hidden_states.shape[0] != sum(split_sizes):
                logger.warning_once(f"KD Token Mismatch: Output {hidden_states.shape[0]}, Expected {sum(split_sizes)}. Skipping KD.")
                return sft_loss_outputs if return_outputs else sft_loss

            per_image_features = torch.split(hidden_states, split_sizes, dim=0)
            
            # GAP
            pooled_features = [feat.mean(dim=0) for feat in per_image_features]
            student_cls_tokens_all = torch.stack(pooled_features) # (Num_Images, Dim)
            
            # Select subset matching teacher images
            student_cls_for_kd = student_cls_tokens_all[
                teacher_image_indices.to(student_cls_tokens_all.device)
            ]
        except Exception as e:
            logger.warning_once(f"KD Processing Error: {e}")
            return sft_loss_outputs if return_outputs else sft_loss

        if student_cls_for_kd.shape[0] == 0:
            return sft_loss_outputs if return_outputs else sft_loss
        
        # Ensure KD module is on the correct device and dtype
        if self.kd_module.student_projection.weight.dtype != self.model.dtype:
            self.kd_module.to(self.model.dtype)
        if self.kd_module.student_projection.weight.device != self.model.device:
            self.kd_module.to(self.model.device)

        with torch.no_grad():
            # Ensure teacher input matches student dtype (usually bfloat16)
            teacher_pixel_values = teacher_pixel_values.to(self.model.device).to(self.model.dtype)
            
            if self.teacher_model.trunk.patch_embed.proj.weight.dtype != self.model.dtype:
                    self.teacher_model.to(self.model.dtype)

            teacher_features = self.teacher_model.trunk.forward_features(
                teacher_pixel_values
            )
            teacher_cls_for_kd = teacher_features[:, 0, :]
        
        # Safety check
        if student_cls_for_kd.shape[0] != teacher_cls_for_kd.shape[0]:
             logger.warning_once(f"KD Batch Mismatch: Student {student_cls_for_kd.shape[0]} vs Teacher {teacher_cls_for_kd.shape[0]}")
             return sft_loss_outputs if return_outputs else sft_loss

        # Projects
        student_projected = self.kd_module.student_projection(student_cls_for_kd)
        # Teacher projection is Identity, but we explicitly call for consistency
        teacher_projected = self.kd_module.teacher_projection(teacher_cls_for_kd.to(self.model.dtype))
        
        kd_loss = self._dino_loss(student_projected, teacher_projected)
        total_loss = sft_loss + self.kd_loss_weight * kd_loss
        print(f"KD Loss: {kd_loss.item()}, SFT Loss: {sft_loss.item()}, Total Loss: {total_loss.item()}")
        # Save losses for logging in the log() method
        self.latest_sft_loss = sft_loss.item()
        self.latest_kd_loss = kd_loss.item()

        if return_outputs:
            return (total_loss,) + sft_loss_outputs[1:]
        else:
            return total_loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Override log to inject SFT and KD losses.
        """
        if hasattr(self, "latest_sft_loss"):
            logs["loss/sft_loss"] = self.latest_sft_loss
        if hasattr(self, "latest_kd_loss"):
            logs["loss/kd_loss"] = self.latest_kd_loss
        super().log(logs)
