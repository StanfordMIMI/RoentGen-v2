import os
import argparse
from PIL import Image
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script to fine-tune unet of the stable diffusion model."
    )
    parser.add_argument(
        "--model_name_or_path",
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
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory when loading models",
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
        "--num_images_per_prompt",
        type=int,
        default=1,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_text_encoder_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=True,
        help="Path to text_encoder model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=256,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--compute_biovil_similarity",
        action="store_true",
        help="Whether to use the attention mask",
    )
    parser.add_argument(
        "--save_similarities_and_embeddings",
        action="store_true",
        help="Whether to use the attention mask",
    )
    parser.add_argument(
        "--use_auth_token",
        type=str,
        default=None,
        help="auth token",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of images.",
    )
    parser.add_argument(
        "--save_similarities_and_embeddings_stem",
        type=str,
        default=None,
        required=False,
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
        "--additional_evaluation_prompt_dir",
        type=str,
        default=None,
        required=False,
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
        "--seed", type=int, default=10, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5,  # TODO can use 1e-05 or 1e-04
        required=True,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,  # TODO can use 1e-05 or 1e-04
        required=True,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--embedding_method",
        type=str,
        default="last_hidden_state",
        required=True,
        help="embedding_method for the encoder",
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
    parser.add_argument(
        "--enforce_tokenizer_max_sentence_length",
        type=int,
        default=None,
        help="enforce_tokenizer_max_sentence_length",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.output_dir is None:
        raise ValueError("You must specify a train data directory.")
    if args.prompt_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.save_similarities_and_embeddings:
        assert args.save_similarities_and_embeddings_stem is not None
        assert args.save_similarities_and_embeddings_stem[-1] != "/"

    # if args.with_prior_preservation:
    #     if args.class_data_dir is None:
    #         raise ValueError("You must specify a data directory for class images.")
    #     if args.class_prompt is None:
    #         raise ValueError("You must specify prompt for class images.")

    return args


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
