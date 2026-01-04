#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# python notebook/eval/eval_RL/eval_mmu.py
python notebook/eval/eval_RL/eval_vqa.py
python notebook/eval/eval_RL/eval_CCRCC.py
python notebook/eval/eval_RL/eval_BreaKHis.py
python notebook/eval/eval_RL/eval_chaoyang.py
python notebook/eval/eval_RL/eval_crc100k.py
python notebook/eval/eval_RL/eval_CRC_MSI.py
python notebook/eval/eval_RL/eval_PanCancer_TIL.py
