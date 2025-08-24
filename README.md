# RoentGen-v2: Improving Performance, Robustness, and Fairness of Radiographic AI Models with Finely-Controllable Synthetic Data

[![Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/stanfordmimi/RoentGen-v2)  [![License](https://img.shields.io/github/license/stanfordmimi/RoentGen-v2?style=for-the-badge)](LICENSE)

⏳ Code and instruction upload in progress...

## 🚀 Inference Instructions

```python
from diffusers import DiffusionPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = DiffusionPipeline.from_pretrained("stanfordmimi/RoentGen-v2")
pipe = pipe.to(device)

prompt = "50 year old White female. Normal chest radiograph."
image = pipe(prompt).images[0]
```

### Large-scale Inference
To run large-scale multi-gpu distributed inference, use the following commands.

Only inference, no quality check:
```bash
accelerate launch --num_processes=1 --mixed_precision bf16 \
 roentgenv2/inference_code/run_inference.py \
 --config_file="./configs/infer_config_demo.yaml"
```

Multi-gpu option:
```bash
accelerate launch --num_processes=4 --multi-gpu --mixed_precision bf16 \
 roentgenv2/inference_code/run_inference.py \
 --config_file="./configs/infer_config_demo.yaml"
```

Inference plus demographics quality check:
```bash
accelerate launch --num_processes=1 --mixed_precision bf16 \
 roentgenv2/inference_code/run_inference_w_quality_check.py \
 --config_file="./configs/infer_config_demo.yaml"
```

Multi-gpu option:
```bash
accelerate launch --num_processes=4 --multi-gpu --mixed_precision bf16 \
 roentgenv2/inference_code/run_inference_w_quality_check.py \
 --config_file="./configs/infer_config_demo.yaml"
```

## 🔧 Finetuning Instructions

In order to finetune RoentGen-v2 on **your own dataset**, follow the instructions below (requires Python 3.9 and cloning the repository).
```bash
accelerate launch --num_processes=1 --mixed_precision bf16 \
 roentgenv2/train_code/train.py \
 --config_file="./configs/train_config_demo.yaml"
```

Multi-gpu option:
```bash
accelerate launch --num_processes 4 --multi_gpu --mixed_precision bf16 \
 roentgenv2/train_code/train.py \
 --config_file="./configs/train_config_demo.yaml"
```
