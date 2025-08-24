# RoentGen-v2: Improving Performance, Robustness, and Fairness of Radiographic AI Models with Finely-Controllable Synthetic Data

[![Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/stanfordmimi/RoentGen-v2)  [![License](https://img.shields.io/github/license/stanfordmimi/RoentGen-v2?style=for-the-badge)](LICENSE)

## 🧨Inference with diffusers

```python
import torch
from diffusers import DiffusionPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = DiffusionPipeline.from_pretrained("stanfordmimi/RoentGen-v2")
pipe = pipe.to(device)

prompt = "50 year old female. Normal chest radiograph."
image = pipe(prompt).images[0]
```

## 🩻 Synthetic CXR Dataset
To be released soon, stay tuned.

![Visuals](assets/visual_examples.png)

## 🚀 Developer Mode
### Environment Setup
Strongly recommended to create a dedicated virtual environment for this project. 
A `requirements.txt` is provided. 
After you install the requirements via your package manager, it is important to run `pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 xformers --index-url https://download.pytorch.org/whl/cu126`. This downgrades torch in order to ensure version compatibility with `monai` and `torchxrayvision` packages. Tested and confirmed to work with wheels for `cu126`, `cu121` and `cu118`.

### Large-scale Inference
To run large-scale multi-gpu distributed inference, use the following commands.

Only inference, no quality check:
```bash
accelerate launch --num_processes=1 --mixed_precision bf16 \
 roentgenv2/inference_code/run_inference.py \
 --config_file="./configs/infer_config_demo.yaml"
```

Only inference, no quality check (multi-gpu):
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

Inference plus demographics quality check (multi-gpu):
```bash
accelerate launch --num_processes=4 --multi-gpu --mixed_precision bf16 \
 roentgenv2/inference_code/run_inference_w_quality_check.py \
 --config_file="./configs/infer_config_demo.yaml"
```

### Finetuning Instructions

In order to finetune RoentGen-v2 on your own dataset, follow the instructions below.
```bash
accelerate launch --num_processes=1 --mixed_precision bf16 \
 roentgenv2/train_code/train.py \
 --config_file="./configs/train_config_demo.yaml"
```

Finetuning (multi-gpu):
```bash
accelerate launch --num_processes 4 --multi_gpu --mixed_precision bf16 \
 roentgenv2/train_code/train.py \
 --config_file="./configs/train_config_demo.yaml"
```
