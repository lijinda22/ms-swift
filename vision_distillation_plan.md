# 下面是修改简述:
swift\llm\argument\train_args.py
swift\llm\train\sft.py
swift\trainers\trainers.py
主要是添加了KD知识蒸馏相关的实现


# 下面是修改的详细描述, 不一定完全准确
详细代码修改方案
步骤 1：修改 swift/llm/argument/train_args.py
在 TrainArguments dataclass 的末尾（__post_init__ 函数之前）添加KD参数：

Python

# swift/llm/argument/train_args.py

# ... (imports) ...
# ... (Seq2SeqTrainingOverrideArguments, SwanlabArguments) ...

@dataclass
class TrainArguments(SwanlabArguments, TunerArguments, BaseArguments, Seq2SeqTrainingOverrideArguments):
    # ... (原有参数, 例如 early_stop_interval) ...
    early_stop_interval: Optional[int] = None

    # --- Knowledge Distillation (KD) Arguments ---
    kd_teacher_model_type: Optional[str] = field(
        default=None,
        metadata={'help': 'Type of the teacher model for ViT KD (e.g., "conchv1_5")'}
    )
    kd_teacher_model_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to the pretrained weights of the teacher model'}
    )
    kd_loss_weight: float = field(
        default=1.0,
        metadata={'help': 'Weight for the knowledge distillation loss (w * KD_Loss)'}
    )
    kd_projection_dim: Optional[int] = field(
        default=None,
        metadata={'help': 'Projection dimension for CLS token alignment. Defaults to student ViT hidden size.'}
    )
    kd_student_T: float = field(
        default=0.1,
        metadata={'help': 'Temperature for student CLS token in DINO loss'}
    )
    kd_teacher_T: float = field(
        default=0.05,
        metadata={'help': 'Temperature for teacher CLS token in DINO loss'}
    )
    kd_center_momentum: float = field(
        default=0.9,
        metadata={'help': 'Momentum for DINO loss center update'}
    )
    # -------------------------------------------------

    def _check_padding_free(self):
        # ... (原有函数) ...
步骤 2：修改 swift/trainers/trainers.py
在文件顶部添加 conchv1_5 的导入和 F：

Python

# swift/trainers/trainers.py
# ... (existing imports) ...
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ... (logger) ...

# 尝试导入教师模型
try:
    # 假设 conchv1_5.py 在您的 python 路径中
    from conchv1_5 import create_model_from_pretrained
except ImportError:
    logger.warning("Could not import create_model_from_pretrained from conchv1_5.py. KD will be disabled if used.")
    create_model_from_pretrained = None
在文件末尾，Seq2SeqTrainer 类的 后面，添加新的 SftKdTrainer 类：

Python

# swift/trainers/trainers.py

# ... (Seq2SeqTrainer 类的定义) ...


class SftKdTrainer(Seq2SeqTrainer):
    """
    Trainer for SFT + ViT Knowledge Distillation.
    Inherits from Seq2SeqTrainer.
    """
    def __init__(self, 
                 *args,
                 sft_args: 'TrainArguments',  # 接收来自 sft.py 的完整 TrainArguments
                 teacher_model: nn.Module, 
                 teacher_transform: Callable, 
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.sft_args = sft_args
        self.teacher_model = teacher_model.eval() # 确保教师模型在评估模式
        self.teacher_transform = teacher_transform
        
        self.kd_loss_weight = sft_args.kd_loss_weight
        self.student_T = sft_args.kd_student_T
        self.teacher_T = sft_args.kd_teacher_T
        self.center_momentum = sft_args.kd_center_momentum
        
        # 缓冲区
        self.student_cls_token_buffer = None

        # 包装 data_collator
        self.original_data_collator = self.data_collator
        self.data_collator = self._kd_data_collator
        
        # 初始化投影层和 DINO 中心
        self._init_kd_components()

    def _kd_data_collator(self, batch: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Wraps the original data collator to extract PIL images
        and prepare teacher_pixel_values.
        """
        
        pil_images_for_teacher = []
        # `batch` 是一个 List[Dict]，每个 Dict 包含 'images': [PIL.Image] 或 []
        for d in batch:
            if d.get('images'):
                pil_images_for_teacher.append(d['images'][0].convert('RGB'))
            else:
                pil_images_for_teacher.append(None) # 占位

        # 调用原始 collator (例如 QwenVLTemplate.data_collator)
        # 这会为学生模型生成 'pixel_values'
        student_inputs = self.original_data_collator(batch, **kwargs)

        teacher_pixel_values_list = []
        valid_image_indices = [] # 记录批次中哪些样本有图像

        for i, pil_img in enumerate(pil_images_for_teacher):
            if pil_img:
                try:
                    teacher_pixel_values_list.append(self.teacher_transform(pil_img))
                    valid_image_indices.append(i)
                except Exception as e:
                    logger.warning(f"Failed to transform image {i} for teacher: {e}")

        if teacher_pixel_values_list:
            # 这些是教师模型的输入
            teacher_pixel_values = torch.stack(teacher_pixel_values_list)
            student_inputs['teacher_pixel_values'] = teacher_pixel_values
            # 告诉 compute_loss 这些图像对应批次中的哪些索引
            student_inputs['teacher_image_indices'] = torch.tensor(valid_image_indices, dtype=torch.long)
        
        return student_inputs

    def _student_hook(self, module, input, output):
        """
        Forward hook to capture the student ViT's [CLS] token.
        """
        if hasattr(output, 'last_hidden_state'):
            # Qwen-VL ViT 输出 (BatchSize, SeqLen, HiddenSize)
            # [CLS] token 是第一个
            self.student_cls_token_buffer = output.last_hidden_state[:, 0, :]
        else:
            logger.warning_once(f"Student ViT hook output type {type(output)} unexpected. Cannot capture CLS token.")

    def _init_kd_components(self):
        # 1. 动态注册钩子到学生ViT
        try:
            # Qwen-VL 的 ViT 模块路径
            student_vit_module = self.model.model.vision_tower.vision_tower
            student_hidden_size = student_vit_module.config.hidden_size
            
            student_vit_module.register_forward_hook(self._student_hook)
            logger.info(f"Registered forward hook on student ViT module: {student_vit_module.__class__.__name__}")
            
        except AttributeError as e:
            logger.error(f"Failed to find student ViT (model.model.vision_tower.vision_tower). KD is disabled. Error: {e}")
            self.kd_loss_weight = 0.0
            return

        # 2. 获取教师ViT的隐藏维度 (来自 conchv1_5.py)
        try:
            teacher_hidden_size = self.teacher_model.trunk.embed_dim
        except AttributeError as e:
            logger.error(f"Failed to get teacher ViT hidden size (teacher_model.trunk.embed_dim). KD is disabled. Error: {e}")
            self.kd_loss_weight = 0.0
            return

        # 3. 确定对齐维度
        projection_dim = self.sft_args.kd_projection_dim
        if projection_dim is None:
            projection_dim = student_hidden_size
            logger.info(f"kd_projection_dim not set. Defaulting to student ViT hidden size: {projection_dim}")

        # 4. 创建投影层
        self.student_projection = nn.Linear(student_hidden_size, projection_dim).to(self.model.device)
        self.teacher_projection = nn.Linear(teacher_hidden_size, projection_dim).to(self.model.device)
        self.teacher_projection.requires_grad_(False) # 冻结
        
        # 5. DINO 损失中心 (Center)
        self.register_buffer("center", torch.zeros(1, projection_dim).to(self.model.device))
        logger.info("Initialized Knowledge Distillation Components.")
        logger.info(f"  Student ViT CLS: {student_hidden_size} -> {projection_dim}")
        logger.info(f"  Teacher ViT CLS: {teacher_hidden_size} -> {projection_dim} (Frozen)")

    def _dino_loss(self, student_output, teacher_output):
        student_out = F.softmax(student_output / self.student_T, dim=-1)
        teacher_out = (teacher_output - self.center) / self.teacher_T
        teacher_out = F.softmax(teacher_out, dim=-1).detach()

        kd_loss = - (teacher_out * torch.log(student_out + 1e-9)).sum(dim=-1).mean()
        
        if self.training:
            with torch.no_grad():
                batch_center = teacher_output.mean(dim=0, keepdim=True)
                self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        return kd_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写 compute_loss:
        1. 弹出教师输入
        2. 计算 SFT 损失 (这会触发 hook)
        3. 计算 Teacher [CLS]
        4. 对齐 Student [CLS] 和 Teacher [CLS]
        5. 计算 KD 损失
        """
        
        teacher_pixel_values = inputs.pop('teacher_pixel_values', None)
        teacher_image_indices = inputs.pop('teacher_image_indices', None)

        # 1. 计算 SFT 损失 (调用 Seq2SeqTrainer.compute_loss)
        #    这会触发 _student_hook, 填充 self.student_cls_token_buffer
        self.student_cls_token_buffer = None # 清空
        sft_loss_outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        sft_loss = sft_loss_outputs.loss

        # 2. 检查KD是否可行
        if self.kd_loss_weight == 0.0 or teacher_pixel_values is None or self.student_cls_token_buffer is None:
            if self.kd_loss_weight > 0.0 and teacher_pixel_values is None:
                logger.warning_once("No valid teacher images in batch. Skipping KD loss.")
            if self.kd_loss_weight > 0.0 and self.student_cls_token_buffer is None:
                logger.warning_once("Student ViT hook did not run (no images in batch?). Skipping KD loss.")
            return sft_loss_outputs if return_outputs else sft_loss
        
        try:
            # 3. 获取学生 [CLS]
            # Qwen-VL collator 会 padding，因此 ViT 总是处理 B 个图像
            # student_cls_token_buffer 的形状是 (B, D_stu)
            student_cls_tokens_all = self.student_cls_token_buffer.clone()
            
            # 筛选出我们拥有教师图像的那些学生 [CLS]
            student_cls_for_kd = student_cls_tokens_all[teacher_image_indices.to(student_cls_tokens_all.device)]

            if student_cls_for_kd.shape[0] == 0:
                # 理论上不应发生，因为 teacher_pixel_values 不是 None
                raise ValueError("No matching student samples for KD.")

            # 4. 计算教师 [CLS]
            with torch.no_grad():
                teacher_features = self.teacher_model.trunk.forward_features(teacher_pixel_values.to(self.model.device))
                teacher_cls_for_kd = teacher_features[:, 0, :] # (B_img, D_tea)
            
            assert student_cls_for_kd.shape[0] == teacher_cls_for_kd.shape[0]

            # 5. 计算 KD 损失
            student_projected = self.student_projection(student_cls_for_kd)
            teacher_projected = self.teacher_projection(teacher_cls_for_kd)
            
            kd_loss = self._dino_loss(student_projected, teacher_projected)
            
            total_loss = sft_loss + self.kd_loss_weight * kd_loss
            
            # 记录损失
            if self.is_world_process_zero() and self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                self.log({'loss/kd_loss': kd_loss.item(), 'loss/sft_loss': sft_loss.item()})
                
        except Exception as e:
            logger.error(f"Error calculating KD loss: {e}. Skipping KD loss for this step.")
            total_loss = sft_loss

        if return_outputs:
            sft_loss_outputs.loss = total_loss
            return sft_loss_outputs
        else:
            return total_loss

步骤 3：修改 swift/llm/train/sft.py
这是集成的关键，我们将修改 SwiftSft 类。

Python

# swift/llm/train/sft.py

# ... (imports) ...
from swift.trainers import TrainerFactory
# --- 添加以下 imports ---
from swift.trainers.trainers import Seq2SeqTrainer  # 导入基类以供检查
try:
    from conchv1_5 import create_model_from_pretrained
except ImportError:
    logger.warning("Could not import create_model_from_pretrained from conchv1_5.py. KD will be disabled if used.")
    create_model_from_pretrained = None
# --- 结束添加 ---

from ..argument import TrainArguments
# ... (imports) ...

logger = get_logger()


class SwiftSft(SwiftPipeline, TunerMixin):
    args_class = TrainArguments
    args: args_class

    def __init__(self, args: Optional[Union[List[str], TrainArguments]] = None) -> None:
        super().__init__(args)
        self.train_msg = {}
        # --- 添加 teacher model 属性 ---
        self.teacher_model = None
        self.teacher_transform = None
        # --- 结束添加 ---
        self._prepare_model_tokenizer()
        self._prepare_template()
        self._prepare_callbacks()
        if self.args.use_flash_ckpt:
            # ... (flash_ckpt logic) ...

    # ... (_prepare_generation_config) ...

    def _prepare_model_tokenizer(self, **kwargs):
        args = self.args
        self.model, self.processor = args.get_model_processor(**kwargs)
        if args.sequence_parallel_size > 1:
            # ... (sequence_parallel logic) ...
        if self.model is None:
            return
        if hasattr(self.model, "hf_device_map"):
            logger.info(f"model.hf_device_map: {self.model.hf_device_map}")

        logger.info(f"model_info: {self.model.model_info}")

        self._prepare_generation_config()

        # --- 在函数末尾添加 KD 教师模型设置 ---
        if args.kd_teacher_model_type:
            if args.kd_teacher_model_type == 'conchv1_5':
                if create_model_from_pretrained is None:
                    raise ImportError(
                        "KD teacher 'conchv1_5' requested, but 'conchv1_5.py' could not be imported. "
                        "Please ensure it is in your PYTHONPATH.")
                if args.kd_teacher_model_path is None:
                    raise ValueError("kd_teacher_model_path must be set for conchv1_5 KD.")
                
                logger.info(f"Loading KD teacher model: {args.kd_teacher_model_type} from {args.kd_teacher_model_path}")
                
                self.teacher_model, self.teacher_transform = create_model_from_pretrained(
                    checkpoint_path=args.kd_teacher_model_path
                )
                # 移到与学生模型相同的设备
                self.teacher_model = self.teacher_model.to(self.model.device).eval()
                self.teacher_model.requires_grad_(False) # 再次确保冻结
                
            else:
                raise NotImplementedError(f"KD teacher model type '{args.kd_teacher_model_type}' not supported.")
        # --- 结束添加 ---

    # ... (_prepare_template, _get_dataset, _get_data_collator, ...) ...
    # ... (直到 run 方法) ...

    def run(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()

        if args.task_type == "seq_cls":
            # ... (seq_cls logic) ...
        args.save_args()

        data_collator = self._get_data_collator()
        # ... (prepare_model, model_parameter_info) ...
        
        trainer_cls = TrainerFactory.get_trainer_cls(args)

        # --- 添加: 覆盖 Trainer 类 (如果KD已启用) ---
        if self.teacher_model is not None:
            try:
                from swift.trainers.trainers import SftKdTrainer
            except ImportError:
                 raise ImportError("Could not import SftKdTrainer. Did you modify swift/trainers/trainers.py?")
                 
            if trainer_cls is not Seq2SeqTrainer:
                logger.warning(
                    f"KD is enabled, but default trainer is {trainer_cls}, not Seq2SeqTrainer. "
                    f"Will use SftKdTrainer, but this might indicate an unexpected setup."
                )
            trainer_cls = SftKdTrainer
            logger.info("Using SftKdTrainer for ViT Knowledge Distillation.")
        # --- 结束添加 ---

        trainer = trainer_cls(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
            **self._get_trainer_kwargs(),  # <--- 修改这里
        )
        return self.train(trainer)

    def _get_trainer_kwargs(self):
        # --- 修改此函数 ---
        # 注入完整的 sft_args (TrainArguments) 和教师模型
        kwargs = {'sft_args': self.args}
        if self.teacher_model is not None:
            kwargs['teacher_model'] = self.teacher_model
            kwargs['teacher_transform'] = self.teacher_transform
        return kwargs
        # --- 结束修改 ---

    def _save_trainer_state(self, trainer):
        # ... (原有函数) ...

    def train(self, trainer):
        # ... (原有函数) ...

# ... (sft_main 函数) ...