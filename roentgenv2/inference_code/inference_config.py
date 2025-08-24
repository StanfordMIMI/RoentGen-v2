from dataclasses import dataclass
import yaml
import argparse

@dataclass
class InferenceConfig:
    use_auth_token: str
    num_images_per_prompt: int
    test_batch_size: int
    embedding_method: str
    guidance_scale: int
    num_inference_steps: int
    mixed_precision: str
    pretrained_text_encoder_name_or_path: str
    enforce_tokenizer_max_sentence_length: int
    model_name_or_path: str
    output_dir: str
    cache_dir: str = None
    prompt_dir: str = None
    data_filter_file: str = None
    seed: int = 10
    revision: str = None
    data_filter_split_token: str = "\n"
    resolution: int = 512
    save_similarities_and_embeddings: bool = False
    additional_evaluation_prompt_dir: str = None
    resume_from_latest: bool = False
    use_metadata_prompt: bool = False
    url_root: str = None
    logging_dir: str = None
    report_to: str = "wandb"
    csv_file_prompts: str = None
    csv_demographics_check: str = None
    sex_model_path: str = None
    retry_limit: int = 1


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
    my_args = InferenceConfig(**config)
    return my_args
