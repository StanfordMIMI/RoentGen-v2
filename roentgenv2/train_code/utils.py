import argparse
import os
from typing import Iterable, Optional
import torch


def parse_args_train():
    parser = argparse.ArgumentParser(
        description="Training script to fine-tune unet of the stable diffusion model."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_text_encoder_name_or_path",
        type=str,
        default=None,
        help="Path to text_encoder model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_auth_token",
        type=str,
        default=None,
        help="auth token",
    )
    parser.add_argument(
        "--embedding_method",
        type=str,
        default="last_hidden_state",
        required=True,
        help="embedding_method for the encoder",
    )
    parser.add_argument(
        "--use_attention_mask",
        action="store_true",
        help="Whether to use the attention mask",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory when loading models",
    )
    parser.add_argument(
        "--random_unet",
        action="store_true",
        help="Whether to initialize the unet randomly",
    )
    parser.add_argument(
        "--enforce_tokenizer_max_sentence_length",
        type=int,
        default=None,
        help="enforce_tokenizer_max_sentence_length",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of images.",
    )
    parser.add_argument(
        "--image_type",
        type=str,
        default="pt",
        required=False,
        choices=["pt", "parameters"],
        help="A folder containing the training data of images.",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of prompts.",
    )
    parser.add_argument(
        "--data_filter_file",
        type=str,
        default=None,
        help="A file filtering based on file names.",
    )
    parser.add_argument(
        "--data_filter_split_token",
        type=str,
        default="\n",
        help="A file filtering based on file names.",
    )
    parser.add_argument(
        "--loss_weights_file",
        type=str,
        default=None,
        help="A file for the loss weights.",
    )
    parser.add_argument(
        "--loss_weights_split_token",
        type=str,
        default="\n",
        help="A file for the loss weights.",
    )
    parser.add_argument(
        "--inference_prompt_file",
        type=str,
        default=None,
        help="A file filtering based on file names.",
    )
    parser.add_argument(
        "--inference_prompt_split_token",
        type=str,
        default="\n",
        help="A file filtering based on file names.",
    )
    parser.add_argument(
        "--inference_prompt_number_per_prompt",
        type=int,
        default=4,
        help="A file filtering based on file names.",
    )
    parser.add_argument(
        "--inference_prompt_output_file",
        type=str,
        default=None,
        help="A file filtering based on file names.",
    )
    parser.add_argument(
        "--save_only_modified_weights",
        action="store_true",
        help="save only the weights that were modified instead of the entire pipeline",
    )
    parser.add_argument(
        "--do_not_save_weights",
        action="store_true",
        help="save only the weights that were modified instead of the entire pipeline",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fine-tune-unet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=10, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,  # TODO can use 1e-05 or 1e-04
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ', `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.image_dir is None:
        raise ValueError("You must specify a train data directory.")
    if args.prompt_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


def forward_with_embedding_method(
    model,
    embedding_method,
    input_ids,
    attention_mask=None,
    position_ids=None,
) -> torch.Tensor:

    if embedding_method == "last_hidden_state_mean":
        return (
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            .last_hidden_state[:, 1:-1, :]
            .mean(axis=1)
        )

    elif embedding_method == "last_hidden_state_cls":
        # Goto method
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).last_hidden_state[:, 0, :]

    elif embedding_method == "last_hidden_state":
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).last_hidden_state

    elif embedding_method[:-1] == "hidden_state_numbered_from_the_end_":
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).hidden_states[-1 * int(embedding_method[-1])]

    elif embedding_method == "pooler_output":
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).pooler_output

    elif embedding_method == "raw_output":
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    elif embedding_method == "get_projected_text_embeddings":
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).get_projected_text_embeddings(**inputs)

    elif embedding_method == "get_text_features":
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).get_text_features(**inputs)

    else:
        raise ValueError("embedding_method not handled")


def forward_self(
    self,
    input_ids,
    embedding_method="last_hidden_state",
    attention_mask=None,
    position_ids=None,
) -> torch.Tensor:

    if embedding_method == "last_hidden_state_mean":
        return (
            self.forward_original(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            .last_hidden_state[:, 1:-1, :]
            .mean(axis=1)
        )

    elif embedding_method == "last_hidden_state_cls":
        # Goto method
        return self.forward_original(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).last_hidden_state[:, 0, :]

    elif embedding_method == "last_hidden_state":
        return self.forward_original(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).last_hidden_state

    elif embedding_method[:-1] == "hidden_state_numbered_from_the_end_":
        return self.forward_original(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).hidden_states[-1 * int(embedding_method[-1])]

    elif embedding_method == "pooler_output":
        return self.forward_original(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).pooler_output

    elif embedding_method == "raw_output":
        return self.forward_original(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    # elif embedding_method == "get_projected_text_embeddings":
    #     return self.forward_original(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #     ).get_projected_text_embeddings(**inputs)

    # elif embedding_method == "get_text_features":
    #     return self.forward_original(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #     ).get_text_features(**inputs)

    else:
        raise ValueError("embedding_method not handled")


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            (
                p.to(device=device, dtype=dtype)
                if p.is_floating_point()
                else p.to(device=device)
            )
            for p in self.shadow_params
        ]
