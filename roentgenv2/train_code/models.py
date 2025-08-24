import itertools
from typing import Iterable, Optional

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    CLIPTextConfig,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPFeatureExtractor,
)

from diffusers.utils.import_utils import is_xformers_available


##########################################################
def load_models(args, accelerator, logger):

    kwargs_from_pretrained = {}
    if args.cache_dir is not None:
        kwargs_from_pretrained["cache_dir"] = args.cache_dir
        kwargs_from_pretrained["revision"] = args.revision

    # Text encoder name to use (pretrained if specified, or default same as model name)
    text_encoder_name = (
        args.pretrained_text_encoder_name_or_path
        if args.pretrained_text_encoder_name_or_path is not None
        else args.pretrained_model_name_or_path
    )

    # Model classes
    if text_encoder_name in [
        "CompVis/stable-diffusion-v1-4",
        "stabilityai/stable-diffusion-2",
        "stabilityai/stable-diffusion-2-base",
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-2-1-base",
    ]:
        text_encoder_config_class = CLIPTextConfig
        tokenizer_class = CLIPTokenizer
        text_encoder_model_class = CLIPTextModel
        freeze_pooler = False
        tokenizer_subfolder_to_use = "tokenizer"
        model_subfolder_to_use = "text_encoder"
    else:
        text_encoder_config_class = AutoConfig
        tokenizer_class = AutoTokenizer
        text_encoder_model_class = AutoModel
        freeze_pooler = True
        tokenizer_subfolder_to_use = None
        model_subfolder_to_use = None

    # Get text encoder config
    if model_subfolder_to_use is not None:
        text_encoder_config = text_encoder_config_class.from_pretrained(
            text_encoder_name,
            subfolder=model_subfolder_to_use,
            trust_remote_code=True,
            **kwargs_from_pretrained,
        )
    else:
        text_encoder_config = text_encoder_config_class.from_pretrained(
            text_encoder_name,
            trust_remote_code=True,
            **kwargs_from_pretrained,
        )

    # CHECK: which case is this?
    if args.embedding_method != "last_hidden_state" and accelerator.is_main_process:
        logger.info(
            "Careful, there might be compatibilities of your selected embedding method with the stable diffusion pipeline"
        )
    # CHECK: which case is this?
    if args.embedding_method[:-1] == "hidden_state_numbered_from_the_end_":
        assert args.embedding_method[-1].isnumeric()
        text_encoder_config.output_hidden_states = True

    # Load the text encoder
    if model_subfolder_to_use is not None:
        text_encoder = text_encoder_model_class.from_pretrained(
            text_encoder_name,
            config=text_encoder_config,
            subfolder=model_subfolder_to_use,
            trust_remote_code=True,
            **kwargs_from_pretrained,
        )
    else:
        text_encoder = text_encoder_model_class.from_pretrained(
            text_encoder_name,
            config=text_encoder_config,
            trust_remote_code=True,
            **kwargs_from_pretrained,
        )

    # Load the tokenizer
    tokenizer_model_max_length = (
        args.enforce_tokenizer_max_sentence_length
        if args.enforce_tokenizer_max_sentence_length is not None
        else (
            text_encoder_config.max_position_embeddings
            if hasattr(text_encoder_config, "max_position_embeddings")
            else None
        )
    )
    if tokenizer_subfolder_to_use is not None:
        tokenizer = tokenizer_class.from_pretrained(
            text_encoder_name,
            model_max_length=tokenizer_model_max_length,
            subfolder=tokenizer_subfolder_to_use,
            trust_remote_code=True,
            **kwargs_from_pretrained,
        )
    else:
        tokenizer = tokenizer_class.from_pretrained(
            text_encoder_name,
            model_max_length=tokenizer_model_max_length,
            trust_remote_code=True,
            **kwargs_from_pretrained,
        )

    # Load the vae
    if args.image_type == "pt":
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            **kwargs_from_pretrained,
        )
    else:
        vae = None

    # Get the unet config
    # CHECK: which cases is this block used in?
    unet_config_changed = False
    if text_encoder_name != args.pretrained_model_name_or_path:
        # In this case, we need to check that things match
        unet_config = UNet2DConditionModel.load_config(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            **kwargs_from_pretrained,
        )
        if text_encoder_config.hidden_size != unet_config["cross_attention_dim"]:
            unet_config_changed = True

            if accelerator.is_main_process:
                logger.info(
                    f"different hidden size {text_encoder_config.hidden_size} {unet_config['cross_attention_dim']}"
                )
                logger.info(
                    "Unet config will be updated to match the text encoder hidden size"
                )

            unet_config["cross_attention_dim"] = text_encoder_config.hidden_size

        if unet_config_changed:
            if not args.random_unet:
                if accelerator.is_main_process:
                    logger.info(
                        "you did not choose to start unet randomly, rectifying!"
                    )
                args.random_unet = True
    else:
        unet_config = None

    # Load the unet
    if args.random_unet:
        if accelerator.is_main_process:
            logger.info("Initializing unet randomly")

        if unet_config_changed:
            if accelerator.is_main_process:
                logger.info("init unet from config file")
            unet = UNet2DConditionModel.from_config(
                unet_config,
                **kwargs_from_pretrained,
            )
        else:
            if accelerator.is_main_process:
                logger.info("init unet from online")
            unet = UNet2DConditionModel.from_config(
                args.pretrained_model_name_or_path,
                subfolder="unet",
                **kwargs_from_pretrained,
            )
    else:
        assert not unet_config_changed
        if accelerator.is_main_process:
            logger.info("Initializing unet with saved weights")
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            **kwargs_from_pretrained,
        )

    # Load the noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        **kwargs_from_pretrained,
    )

    return (
        text_encoder,
        tokenizer,
        vae,
        unet,
        noise_scheduler,
        text_encoder_name,
        freeze_pooler,
        unet_config_changed,
        unet_config,
        kwargs_from_pretrained,
    )


##########################################################
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
