from dataclasses import dataclass, field
from typing import List
from collections import defaultdict
import yaml
import argparse


@dataclass
class BaseConfig:
    """All the configurable parameters for the training script."""

    # Path to pretrained model or model identifier from huggingface.co/models.
    pretrained_model_name_or_path: str = None
    # Revision of pretrained model identifier from huggingface.co/models.
    revision: str = None
    # Path to text_encoder model or model identifier from huggingface.co/models.
    pretrained_text_encoder_name_or_path: str = None
    # Hugging face authentication token
    use_auth_token: str = None
    embedding_method: str = "last_hidden_state"
    use_attention_mask: bool = False
    # Cache directory when loading models
    cache_dir: str = None
    # Whether to initialize the unet with random weights
    random_unet: bool = False
    enforce_tokenizer_max_sentence_length: int = None

    # Whether to load dataset from wds instead of image/prompt directories
    use_wds_dataset: bool = False
    url_root: str = None
    # A folder containing the training data of images.
    image_dir: str = None
    image_type: str = "pt"
    # A folder containing the training data of prompts.
    prompt_dir: str = None
    # A file filtering based on file names.
    data_filter_file: str = None
    data_filter_split_token: str = "\n"
    loss_weights_file: str = None
    loss_weights_split_token: str = "\n"

    inference_prompt_file: str = None
    inference_prompt_split_token: str = "\n"
    inference_prompt_number_per_prompt: int = 4
    inference_prompt_output_file: str = None

    # save only the weights that were modified instead of the entire pipeline
    save_only_modified_weights: bool = False
    do_not_save_weights: bool = False
    # The output directory where the model predictions and checkpoints will be written.
    output_dir: str = None

    # Seed number for reproducible training
    seed: int = 10
    # The resolution for input images, all the images in the train/validation
    # dataset will be resized to this
    resolution: int = 512
    # Whether to center crop images before resizing to resolution
    center_crop: bool = False

    # Whether to train the text encoder
    train_text_encoder: bool = False
    # Batch size (per device) for the training dataloader.
    train_batch_size: int = 4
    # For debugging purposes or quicker training, truncate the number
    # of training examples to this value if set
    max_train_samples: int = None
    num_train_epochs: int = 100
    # Total number of training steps to perform.
    # If provided, overrides num_train_epochs.
    max_train_steps: int = None
    # Number of updates steps to accumulate before performing a backward/update pass.
    gradient_accumulation_steps: int = 1
    # Whether or not to use gradient checkpointing to save memory
    # at the expense of slower backward pass.
    gradient_checkpointing: bool = False

    # Initial learning rate (after the potential warmup period) to use.
    learning_rate: float = 5e-06
    # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
    scale_lr: bool = False
    # Choose between ["linear", "cosine", "cosine_with_restarts",
    # "polynomial", "constant", "constant_with_warmup"]
    lr_scheduler: str = "constant"
    # Number of steps for the warmup in the lr scheduler.
    lr_warmup_steps: int = 500
    # Whether or not to use 8-bit Adam from bitsandbytes.
    use_8bit_adam: bool = False
    # Whether to use EMA model for the unet.
    use_ema: bool = False

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-02
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0

    logging_dir: str = "logs"
    report_to: str = "wandb"

    # Choose between fp16 and bf16 (requires Nvidia Ampere GPU)
    mixed_precision: str = "no"
    # For distributed training: local_rank
    local_rank: int = 1

    # Save a checkpoint of the training state every X updates.
    # Checkpoints can be used for resuming training via `--resume_from_checkpoint`.
    checkpointing_steps: int = 500
    # Max number of checkpoints to store.
    checkpoints_total_limit: int = None
    # Whether training should be resumed from a previous checkpoint. Use a path saved by
    # `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.
    resume_from_checkpoint: str = None

    def get_config(self):
        return self.__dict__


def load_config(file_path):
    with open(file_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as e:
            print("Error loading YAML:", e)
            return None


def get_args_from_config():
    parser = argparse.ArgumentParser(
        description="Training script to fine-tune unet of the stable diffusion model."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        required=True,
        help="Experiment config file.",
    )
    args = parser.parse_args()
    config = load_config(args.config_file)
    my_args = BaseConfig(**config)
    return my_args
