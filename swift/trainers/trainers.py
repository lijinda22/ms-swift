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
        teacher_model: Optional[nn.Module] = None,
        teacher_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sft_args = sft_args
        
        # Handle multiple teachers
        self.teacher_models = []
        # Check if model has attached teachers (from sft.py)
        if hasattr(self.model, "teacher_models") and self.model.teacher_models:
             self.teacher_models = self.model.teacher_models
        elif teacher_model is not None:
             self.teacher_models = [teacher_model.eval()]
        
        # Parse weights
        self.kd_loss_weights = sft_args.kd_loss_weight
        if not isinstance(self.kd_loss_weights, list):
             self.kd_loss_weights = [self.kd_loss_weights] * len(self.teacher_models)
        
        # Ensure weights match teachers
        if len(self.kd_loss_weights) < len(self.teacher_models):
             logger.warning(f"KD weights count ({len(self.kd_loss_weights)}) less than teachers ({len(self.teacher_models)}). Extending with last weight.")
             self.kd_loss_weights.extend([self.kd_loss_weights[-1]] * (len(self.teacher_models) - len(self.kd_loss_weights)))

        self.kd_loss_type = getattr(sft_args, "kd_loss_type", "dino")
        self.kd_token_strategy = getattr(sft_args, "kd_token_strategy", "cls_mean")
        self.kd_weight_strategy = getattr(sft_args, "kd_weight_strategy", "fixed") # fixed or similarity_weighted

        self.student_T = sft_args.kd_student_T
        self.teacher_T = sft_args.kd_teacher_T
        self.center_momentum = sft_args.kd_center_momentum
        self.student_cls_token_buffer = None
        self.student_spatial_merge_size = 2  # Default for Qwen2-VL/3-VL
        self.kd_modules: Optional[nn.ModuleList] = None
        self.original_data_collator = self.data_collator
        self.data_collator = self._kd_data_collator
        self._init_kd_components()

    def _get_teacher_info(self, teacher_model: nn.Module):
        """
        Helper to extract embed_dim, num_prefix_tokens, and forward function.
        Returns: (embed_dim, num_prefix_tokens, forward_fn)
        """
        # Defaults
        num_prefix = 1 # Most ViTs have 1 CLS token
        
        # Case 1: CoCa (Conch) - has .visual.trunk
        if hasattr(teacher_model, 'visual') and hasattr(teacher_model.visual, 'trunk'):
            trunk = teacher_model.visual.trunk
            return teacher_model.embed_dim, getattr(trunk, 'num_prefix_tokens', 1), trunk.forward_features
        
        # Case 2: CONCHVisionTower (Conch V1.5) - has .trunk
        if hasattr(teacher_model, 'trunk'):
             trunk = teacher_model.trunk
             return trunk.embed_dim, getattr(trunk, 'num_prefix_tokens', 1), trunk.forward_features
             
        # Case 3: Standard ViT (UNI, UNI2) - is the ViT itself
        if hasattr(teacher_model, 'embed_dim'):
            if hasattr(teacher_model, 'forward_features'):
                 return teacher_model.embed_dim, getattr(teacher_model, 'num_prefix_tokens', 1), teacher_model.forward_features
            
        # Fallback for timm models
        if hasattr(teacher_model, 'num_features'):
             return teacher_model.num_features, getattr(teacher_model, 'num_prefix_tokens', 1), getattr(teacher_model, 'forward_features', teacher_model)
             
        raise ValueError(f"Unknown teacher model structure: {teacher_model.__class__.__name__}")

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override save_model to also save the KD adapter weights.
        """
        super().save_model(output_dir, _internal_call)
        # Only save on the main process to avoid race conditions and errors on other ranks
        # Only save on the main process to avoid race conditions and errors on other ranks
        if self.is_world_process_zero() and output_dir and self.kd_modules:
            # Ensure the directory exists before saving
            os.makedirs(output_dir, exist_ok=True) # Ensure dir exists
            kd_path = os.path.join(output_dir, "kd_module.pt")
            torch.save(self.kd_modules.state_dict(), kd_path)
            logger.info(f"Saved KD modules to {kd_path}")


    def _kd_data_collator(self, batch: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        KD Data Collator:
        仅负责从 batch 中提取预处理好的 'teacher_pixel_values' 并堆叠 
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

        teacher_input_lists = [[] for _ in range(len(self.teacher_models))]
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
                    # item['teacher_pixel_values'] is List[Tensor], one per teacher
                    t_vals = item['teacher_pixel_values']
                    if len(t_vals) == len(self.teacher_models):
                        for t_idx, val in enumerate(t_vals):
                            teacher_input_lists[t_idx].append(val)
                        # Map to the FIRST image of this sample in the student buffer
                        teacher_indices.append(student_img_idx)
                    else:
                        logger.warning_once(f"Teacher value count mismatch. Expected {len(self.teacher_models)}, got {len(t_vals)}")
                else:
                     # Student has image but Teacher failed/missing.
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

        if teacher_indices:
            # Stack per teacher
            student_inputs["teacher_pixel_values"] = [torch.stack(l) for l in teacher_input_lists]
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
            return

        self.kd_modules = nn.ModuleList()
        
        for i, t_model in enumerate(self.teacher_models):
            try:
                teacher_hidden_size, _, _ = self._get_teacher_info(t_model)
                
                kd_mod = KnowledgeDistillationModule(
                    student_hidden_size=student_hidden_size,
                    teacher_hidden_size=teacher_hidden_size,
                ).to(self.model.device)
                self.kd_modules.append(kd_mod)
                
                logger.info(f"Initialized KD Component for Teacher {i} ({t_model.__class__.__name__}): {student_hidden_size} -> {teacher_hidden_size} ({self.kd_loss_type}, {self.kd_token_strategy})")
                
            except Exception as e:
                logger.error(
                    f"Failed to get teacher ViT hidden size for teacher {i}. Skipping this teacher. Error: {e}"
                )
                self.kd_modules.append(None) # Add placeholder to keep indices aligned
                self.kd_loss_weights[i] = 0.0 # Disable
                continue

        # CRITICAL: Attach to model so optimizer picks up the parameters!
        # We use a distinct name to avoid conflict with existing modules.
        if not hasattr(self.model, "kd_adapters"):
            self.model.kd_adapters = self.kd_modules
        else:
            logger.warning("Model already has 'kd_adapters'. Using existing one (resume?).")
            self.kd_modules = self.model.kd_adapters

    def _dino_loss(self, student_output, teacher_output, kd_adapter):
        # DINO损失计算
        student_out = F.softmax(student_output / self.student_T, dim=-1)
        teacher_out = (teacher_output - kd_adapter.center) / self.teacher_T
        teacher_out = F.softmax(teacher_out, dim=-1).detach()
        kd_loss = -(teacher_out * torch.log(student_out + 1e-9)).sum(dim=-1).mean()
        
        # EMA Center Update (handled here or outside? Usually outside during forward)
        # But for strictly loss calculation, we just read center. 
        # Center update usually happens in the module forward or a separate step.
        # existing implementation didn't update center here? 
        # Ah, the original code likely updated center in forward pass of KD module.
        # But we are manually forwarding projections.
        # We need to ensure center is updated.
        if self.training:
           kd_adapter.update_center(teacher_output)
           
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
            not self.teacher_models
            or self.kd_modules is None
            or teacher_pixel_values is None
            or self.student_cls_token_buffer is None
        ):
            logger.warning_once("KD Error: Missing teacher models, kd_modules, teacher_pixel_values, or student_cls_token_buffer. Skipping KD.")
            return sft_loss_outputs if return_outputs else sft_loss
        
        # --- Student Feature Processing (Split) ---
        # Qwen3-VL/Qwen2.5-VL output is flattened. We need grid_thw to split.
        if image_grid_thw is None:
             logger.warning_once("KD Error: image_grid_thw missing in inputs. Cannot split flattened student features. Skipping KD.")
             return sft_loss_outputs if return_outputs else sft_loss

        hidden_states = self.student_cls_token_buffer
        
        try:
            split_sizes = (image_grid_thw.to(hidden_states.device).prod(dim=-1) // (self.student_spatial_merge_size ** 2)).tolist()
            
            if hidden_states.shape[0] != sum(split_sizes):
                logger.warning_once(f"KD Token Mismatch: Output {hidden_states.shape[0]}, Expected {sum(split_sizes)}. Skipping KD.")
                return sft_loss_outputs if return_outputs else sft_loss

            per_image_features = torch.split(hidden_states, split_sizes, dim=0)
            
            # Sub-Select student features that correspond to teacher images
            teacher_indices_tensor = teacher_image_indices.to(hidden_states.device)
            # list of Tensors
            student_features_subset = [per_image_features[idx] for idx in teacher_indices_tensor]
            
            # --- Loop over teachers ---
            teacher_raw_losses = []
            teacher_sim_scores = []
            
            for i, (t_model, t_pixels, t_adapter) in enumerate(zip(self.teacher_models, teacher_pixel_values, self.kd_modules)):
                
                # Skip invalid/disabled teachers
                if t_adapter is None:
                     teacher_raw_losses.append(torch.tensor(0.0, device=self.model.device))
                     teacher_sim_scores.append(torch.tensor(-100.0, device=self.model.device)) # Low sim for disabled
                     continue

                # Device check for adapter
                if t_adapter.student_projection.weight.dtype != self.model.dtype: t_adapter.to(self.model.dtype)
                if t_adapter.student_projection.weight.device != self.model.device: t_adapter.to(self.model.device)
                
                # Teacher Forward
                # Get correct forward function and handle device/dtype
                try:
                    _, num_prefix_tokens, forward_fn = self._get_teacher_info(t_model)
                    
                    # Ensure teacher is on correct device/dtype (Global check)
                    # Use a robust check or just force cast if needed. 
                    # Assuming checking one parameter is enough proxy.
                    first_param = next(t_model.parameters(), None)
                    if first_param is not None and (first_param.dtype != self.model.dtype or first_param.device != self.model.device):
                         t_model.to(device=self.model.device, dtype=self.model.dtype)

                    t_pixels = t_pixels.to(self.model.device).to(self.model.dtype)
                    
                    with torch.no_grad():
                        t_out = forward_fn(t_pixels) # (B, L_t, D_t)
                except Exception as e:
                    logger.warning_once(f"Teacher {i} forward failure: {e}. Skipping.")
                    teacher_raw_losses.append(torch.tensor(0.0, device=self.model.device))
                    teacher_sim_scores.append(torch.tensor(-100.0, device=self.model.device))
                    continue
                
                loss_sum_for_teacher = 0.0
                sim_sum_for_teacher = 0.0
                valid_sample_count = len(student_features_subset)
                
                # Iterate over batch items (aligned with student_features_subset)
                for k, s_feat in enumerate(student_features_subset):
                    # s_feat: (Student_Tokens, D_s)
                    t_feat = t_out[k] # (Teacher_Tokens, D_t)
                    
                    # --- Compute Similarity for Weighting (Always usage CLS/GAP comparison) ---
                    # Student Mean
                    s_mean = s_feat.mean(dim=0, keepdim=True) # (1, D_s)
                    # Teacher CLS or Global Rep
                    if num_prefix_tokens > 0:
                        t_cls = t_feat[0].unsqueeze(0).to(self.model.dtype) # (1, D_t)
                    else:
                        # Fallback to Mean if no CLS
                        t_cls = t_feat.mean(dim=0, keepdim=True).to(self.model.dtype)

                    # Project Student Mean to Teacher Dim for Sim Calc
                    s_mean_proj = t_adapter.student_projection(s_mean)
                    t_cls_proj = t_adapter.teacher_projection(t_cls)
                    
                    # Cosine Sim
                    sim = F.cosine_similarity(s_mean_proj, t_cls_proj, dim=-1) # (1,)
                    sim_sum_for_teacher += sim
                    
                    # --- Strategy Dispatch for LOSS ---
                    if self.kd_token_strategy == "cls_mean":
                        # Reuse Projected features
                        s_proj = s_mean_proj
                        t_proj = t_cls_proj
                        
                        if self.kd_loss_type == "dino":
                            loss_k = self._dino_loss(s_proj, t_proj, t_adapter)
                        else: # mse
                            # L2 Normalize for stability equivalent to Cosine Distance
                            s_proj = F.normalize(s_proj, p=2, dim=-1)
                            t_proj = F.normalize(t_proj, p=2, dim=-1)
                            loss_k = F.mse_loss(s_proj, t_proj)
                            
                        loss_sum_for_teacher += loss_k
                        
                    elif self.kd_token_strategy == "patch_mse":
                        # Teacher Patch: Interpolate to Student quantity
                        # Teacher: t_feat[num_prefix_tokens:] -> (L_t-prefix, D_t)
                        
                        t_patches_flat = t_feat[num_prefix_tokens:]
                        num_patches = t_patches_flat.shape[0]
                        side = int(num_patches**0.5)
                        if side * side != num_patches:
                            # Not square? Skip or warn
                            continue
                            
                        t_patches = t_patches_flat.reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0) # (1, D_t, H, W)
                        
                        grid_idx = teacher_indices_tensor[k]
                        t, h, w = image_grid_thw[grid_idx]
                        
                        # Student H, W from grid.
                        h_map, w_map = h // self.student_spatial_merge_size, w // self.student_spatial_merge_size
                        
                        # Interpolate Teacher to Student Feature Map Size
                        t_interp = F.interpolate(t_patches.float(), size=(h_map, w_map), mode='bilinear', align_corners=False)
                        t_interp = t_interp.to(t_feat.dtype).squeeze(0).permute(1, 2, 0).reshape(-1, t_feat.shape[-1]) # (h*w, D_t)
                        
                        # Verify shape match
                        if s_feat.shape[0] != t_interp.shape[0]:
                             continue

                        # Project Student to Teacher Dim
                        s_proj = t_adapter.student_projection(s_feat) # (h*w, D_t)
                        t_proj = t_adapter.teacher_projection(t_interp.to(self.model.dtype))
                        
                        # Normalized MSE
                        if self.kd_loss_type == "mse":
                             s_proj = F.normalize(s_proj, p=2, dim=-1)
                             t_proj = F.normalize(t_proj, p=2, dim=-1)

                        if self.kd_loss_type == "dino":
                             loss_k = self._dino_loss(s_proj, t_proj)
                        else:
                             loss_k = F.mse_loss(s_proj, t_proj)

                        loss_sum_for_teacher += loss_k
                
                # Average over batch
                if valid_sample_count > 0:
                     loss_sum_for_teacher /= valid_sample_count
                     sim_sum_for_teacher /= valid_sample_count
                
                teacher_raw_losses.append(loss_sum_for_teacher)
                teacher_sim_scores.append(sim_sum_for_teacher)

            # --- Aggregation ---
            total_kd_loss = 0.0
            
            if self.kd_weight_strategy == "similarity_weighted":
                # Softmax over similarity scores
                sim_tensor = torch.stack(teacher_sim_scores) if teacher_sim_scores else torch.tensor([])
                if sim_tensor.numel() > 0:
                    weights = F.softmax(sim_tensor, dim=0) # [Num_Teachers]
                    # Log weights periodically?
                    # logger.info(f"Dynamic Weights: {weights.tolist()}") 
                    for i, loss in enumerate(teacher_raw_losses):
                        total_kd_loss += weights[i] * loss
                
                # Apply Global Scale
                # In this mode, kd_loss_weight[0] is treated as the global scale
                global_scale = self.kd_loss_weights[0] if self.kd_loss_weights else 0.0
                total_kd_loss *= global_scale
                
            else: # fixed weighting
                for i, loss in enumerate(teacher_raw_losses):
                     total_kd_loss += self.kd_loss_weights[i] * loss

            kd_loss_weighted = total_kd_loss
            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                kd_loss_weighted = kd_loss_weighted / self.args.gradient_accumulation_steps
            
            total_loss = sft_loss + kd_loss_weighted
        
        # Log SFT and KD loss
        if self.model.training:
           sft_loss_scalar = sft_loss.item()
           kd_loss_scalar = total_kd_loss.item() if isinstance(total_kd_loss, torch.Tensor) else total_kd_loss
           if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                # If these losses were already divided by accumulation steps in compute_loss logic (which they are),
                # we need to multiply them back if we want "loss per micro-batch" for averaging.
                # However, MeanMetric averages whatever we feed it. 
                # Standard 'loss' is averaged over steps.
                # If we want to align with standard loss logging, we should feed the "scaled" loss * accumulation_steps?
                # Actually, standard Trainer logs 'tr_loss' which is the running average of loss.

                # Simple approach: Let's log the value of the loss component *as contributing to the gradient*, 
                # but scaled up by accumulation steps so it represents the "per-batch" loss magnitude,
                # which is more intuitive.
                # Note: `sft_loss` from `super().compute_loss` IS already divided by accumulation steps
                # if `num_items_in_batch` is passed. So to get the "actual" loss value (batch average),
                # we multiply by acc_steps.
                sft_loss_scalar *= self.args.gradient_accumulation_steps
                
                # kd_loss is the raw DINO loss, which is what we want to track.
                # The 'kd_loss_weighted' is the one that gets divided by `gradient_accumulation_steps`.
                # We log the raw KD loss (unweighted, unscaled) as 'loss/kd_loss'.
                
           self.custom_metrics["train"]["sft_loss"].update(sft_loss_scalar)
           self.custom_metrics["train"]["kd_loss"].update(kd_loss_scalar)

        if return_outputs:
            return (total_loss,) + sft_loss_outputs[1:]
        else:
            return total_loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Override log to inject SFT and KD losses.
        """
        # Retrieve and reset custom metrics
        if "train" in self.custom_metrics:
            for key, metric in self.custom_metrics["train"].items():
                if key in ["sft_loss", "kd_loss"]:
                    res = metric.compute()
                    if res["value"] is not None:
                         logs[f"loss/{key}"] = res["value"]
                    metric.reset()
        
        super().log(logs)
