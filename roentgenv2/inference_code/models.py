from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    CLIPTextConfig,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPFeatureExtractor,
)

from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)


def get_config_tokenizer_model_classes(diffusion_model_name, model_name):
    config_class = AutoConfig
    tokenizer_class = AutoTokenizer
    model_class = AutoModel

    if (
        model_name == "openai/clip-vit-large-patch14"
        or model_name == "CompVis/stable-diffusion-v1-4"
        or model_name == "stabilityai/stable-diffusion-2"
        or model_name == "stabilityai/stable-diffusion-2-1"
        or model_name == "stabilityai/stable-diffusion-2-1-base"
        or model_name == "stabilityai/stable-diffusion-2-base"
    ):
        config_class = CLIPTextConfig
        tokenizer_class = CLIPTokenizer
        model_class = CLIPTextModel

    tokenizer_subfolder_to_use = "text_encoder_and_tokenizer"
    model_subfolder_to_use = "text_encoder_and_tokenizer"
    if (
        diffusion_model_name == "CompVis/stable-diffusion-v1-4"
        or diffusion_model_name == "stabilityai/stable-diffusion-2"
        or diffusion_model_name == "stabilityai/stable-diffusion-2-1"
        or diffusion_model_name == "stabilityai/stable-diffusion-2-1-base"
        or diffusion_model_name == "stabilityai/stable-diffusion-2-base"
    ) and (
        model_name == "CompVis/stable-diffusion-v1-4"
        or model_name == "stabilityai/stable-diffusion-2"
        or model_name == "stabilityai/stable-diffusion-2-1"
        or model_name == "stabilityai/stable-diffusion-2-1-base"
        or model_name == "stabilityai/stable-diffusion-2-base"
    ):
        tokenizer_subfolder_to_use = "tokenizer"
        model_subfolder_to_use = "text_encoder"

    return (
        config_class,
        tokenizer_class,
        model_class,
        tokenizer_subfolder_to_use,
        model_subfolder_to_use,
    )


def load_models(args):
    use_auth_token = args.use_auth_token
    del args.use_auth_token

    kwargs_from_pretrained = {}
    if args.cache_dir is not None:
        kwargs_from_pretrained["cache_dir"] = args.cache_dir
        kwargs_from_pretrained["revision"] = args.revision

    (
        text_encoder_config_class,
        tokenizer_class,
        text_encoder_model_class,
        tokenizer_subfolder_to_use,
        model_subfolder_to_use,
    ) = get_config_tokenizer_model_classes(
        args.model_name_or_path, args.pretrained_text_encoder_name_or_path
    )

    if model_subfolder_to_use is not None:
        text_encoder_config = text_encoder_config_class.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            subfolder=model_subfolder_to_use,
            use_auth_token=use_auth_token,
            **kwargs_from_pretrained,
        )
    else:
        text_encoder_config = text_encoder_config_class.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            use_auth_token=use_auth_token,
            **kwargs_from_pretrained,
        )

    if args.embedding_method != "last_hidden_state":
        print(
            "Careful, there might be compatibilities of your selected embedding method"
            + "with the stable diffusion pipeline"
        )

    if args.embedding_method[:-1] == "hidden_state_numbered_from_the_end_":
        assert args.embedding_method[-1].isnumeric()
        text_encoder_config.output_hidden_states = True

    if model_subfolder_to_use is not None:
        text_encoder = text_encoder_model_class.from_pretrained(
            args.model_name_or_path,
            config=text_encoder_config,
            trust_remote_code=True,
            subfolder=model_subfolder_to_use,
            use_auth_token=use_auth_token,
            **kwargs_from_pretrained,
        )
    else:
        text_encoder = text_encoder_model_class.from_pretrained(
            args.model_name_or_path,
            config=text_encoder_config,
            trust_remote_code=True,
            use_auth_token=use_auth_token,
            **kwargs_from_pretrained,
        )

    if tokenizer_subfolder_to_use is not None:
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            model_max_length=(
                args.enforce_tokenizer_max_sentence_length
                if args.enforce_tokenizer_max_sentence_length is not None
                else (
                    text_encoder_config.max_position_embeddings
                    if hasattr(text_encoder_config, "max_position_embeddings")
                    else None
                )
            ),
            trust_remote_code=True,
            subfolder=tokenizer_subfolder_to_use,
            use_auth_token=use_auth_token,
            **kwargs_from_pretrained,
        )
    else:
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            model_max_length=(
                args.enforce_tokenizer_max_sentence_length
                if args.enforce_tokenizer_max_sentence_length is not None
                else (
                    text_encoder_config.max_position_embeddings
                    if hasattr(text_encoder_config, "max_position_embeddings")
                    else None
                )
            ),
            trust_remote_code=True,
            use_auth_token=use_auth_token,
            **kwargs_from_pretrained,
        )

    vae = AutoencoderKL.from_pretrained(
        args.model_name_or_path,
        subfolder="vae",
        use_auth_token=use_auth_token,
        **kwargs_from_pretrained,
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.model_name_or_path,
        subfolder="unet",
        use_auth_token=use_auth_token,
        **kwargs_from_pretrained,
    )

    return text_encoder, tokenizer, vae, unet
