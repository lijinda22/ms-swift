# SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning) Project Context

## Project Overview

SWIFT is an official framework provided by the ModelScope community for fine-tuning and deploying large language models and multi-modal large models. It supports the training (pre-training, fine-tuning, human alignment), inference, evaluation, quantization, and deployment of 600+ large models and 300+ multi-modal large models.

Key features include:
- Support for 600+ pure text large models and 300+ multi-modal large models
- Lightweight training techniques (LoRA, QLoRA, Llama-Pro, GaLore, etc.)
- Human alignment training methods (DPO, GRPO, PPO, KTO, CPO, etc.)
- Hardware support for CPU, GPU (RTX series, A10/A100/H100), Ascend NPU, MPS
- Distributed training support (DDP, DeepSpeed ZeRO, FSDP, Megatron)
- Inference acceleration (vLLM, SGLang, LMDeploy)
- Web UI based on Gradio

## Project Structure

```
ms-swift/
├── .dev_scripts/          # Development scripts
├── .github/              # GitHub configurations
├── asset/                # Project assets
├── docs/                 # Documentation
├── examples/             # Example scripts for training, inference, etc.
├── notebook/             # Jupyter notebooks
├── requirements/         # Dependency requirements
├── scripts/              # Utility scripts
├── swift/                # Main source code
│   ├── cli/              # Command-line interface
│   ├── hub/              # Model hub integration
│   ├── llm/              # Large language model components
│   ├── megatron/         # Megatron parallelism support
│   ├── plugin/           # Plugin system
│   ├── trainers/         # Training implementations
│   ├── tuners/           # Model tuning methods
│   ├── ui/               # Web UI components
│   ├── utils/            # Utility functions
│   ├── __init__.py       # Package initialization
│   └── version.py        # Version information
├── tests/                # Test suite
├── README.md            # Main project documentation
├── requirements.txt     # Core dependencies
├── setup.py             # Package setup
└── Makefile             # Build commands
```

## Setup and Installation

The project can be installed in two ways:

1. Using pip:
   ```bash
   pip install ms-swift -U
   ```

2. From source:
   ```bash
   git clone https://github.com/modelscope/ms-swift.git
   cd ms-swift
   pip install -e .
   ```

Python 3.10+ is recommended, with PyTorch >=2.0 required. The project has a comprehensive set of dependencies managed through requirements.txt and related files in the requirements/ directory.

## Building and Running

### CLI Usage
The framework provides a command-line interface with several commands:
- `swift sft` - Supervised fine-tuning
- `swift pt` - Pre-training
- `swift infer` - Inference
- `swift web-ui` - Launch web interface
- `swift export` - Export models
- `swift eval` - Evaluate models

### Example Training Command
```bash
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 16 \
    --output_dir output
```

### Web UI
The framework includes a Gradio-based Web UI for zero-threshold training and deployment:
```bash
SWIFT_UI_LANG=en swift web-ui
```

## Key Technologies and Training Methods

The framework supports various training methods:
- Lightweight fine-tuning: LoRA, QLoRA, DoRA, LoRA+, ReFT, LLaMAPro, GaLore
- Human alignment: DPO, GRPO, RM, PPO, GKD, KTO, CPO, SimPO, ORPO
- Quantization: BNB, AWQ, GPTQ, AQLM, HQQ, EETQ
- Multi-modal training for images, videos, and audio

## Development Conventions

- Python 3.10+ is the recommended version
- The codebase uses PyTorch and HuggingFace Transformers as core dependencies
- PEFT library is used for parameter-efficient fine-tuning techniques
- The project follows standard Python packaging conventions
- CLI commands are implemented in the swift/cli module
- Model tuning methods are implemented in the swift/tuners module
- Training logic is handled by the swift/trainers module

## Testing and Quality

The project includes a test suite in the tests/ directory and uses standard Python testing practices. A Makefile provides commands for building wheels, testing, linting, and cleaning up build artifacts.