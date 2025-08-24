import os
import json
import random
from PIL import Image

import transformers
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker


def create_and_save_pipeline(
    args,
    accelerator,
    unet,
    text_encoder,
    text_encoder_name,
    tokenizer,
    vae,
    ema_unet,
    kwargs_from_pretrained,
    unet_config_changed,
    unet_config,
):
    # Unwrap unet
    unet = accelerator.unwrap_model(unet)
    if args.use_ema:
        ema_unet.copy_to(unet.parameters())

    # CHECK: do I need the third condition in this if?
    if (
        (text_encoder_name == args.pretrained_model_name_or_path)
        and (type(tokenizer) == transformers.tokenization_utils.PreTrainedTokenizer)
        and (text_encoder_name[:11] != "save_model_")
    ):
        assert not unet_config_changed
        # Unwrap text encoder
        if args.train_text_encoder:
            text_encoder = accelerator.unwrap_model(text_encoder)
        # Build pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            unet=unet,
            tokenizer=tokenizer,
            **kwargs_from_pretrained,
        )
    else:  # CHECK: which case is this?
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            **kwargs_from_pretrained,
        )
        args.inference_prompt_file = None

    if (not args.do_not_save_weights) and (not args.save_only_modified_weights):
        # Save the pipeline
        pipeline.save_pretrained(args.output_dir)

        # Save unet
        if unet_config_changed:
            # save possible modification to the unet config file
            print("saved different unet config")

            with open(
                os.path.join(args.output_dir, "unet", "config.json"), "w"
            ) as outfile:
                json.dump(unet_config, outfile)
        # Save text encoder
        if args.train_text_encoder:
            text_encoder = accelerator.unwrap_model(text_encoder)
        text_encoder.save_pretrained(
            os.path.join(args.output_dir, "text_encoder_and_tokenizer")
        )
        # Save tokenizer
        tokenizer.save_pretrained(
            os.path.join(args.output_dir, "text_encoder_and_tokenizer")
        )
    # CHECK: do i still need this?
    elif (not args.do_not_save_weights) and (args.save_only_modified_weights):
        raise Warning("might be broken")
        unet.save_pretrained(os.path.join(args.output_dir, "unet"))
        if args.train_text_encoder:
            accelerator.unwrap_model(text_encoder).save_pretrained(
                os.path.join(args.output_dir, "text_encoder")
            )

    # TODO: figure out inference use case and clean up
    if args.inference_prompt_file is not None:

        def dummy(images, **kwargs):
            return images, [False] * len(images)

        print(pipeline.safety_checker)
        if pipeline.safety_checker is not None:
            assert pipeline.feature_extractor is not None
            pipeline.safety_checker = dummy

        def image_grid(imgs, rows, cols):
            assert len(imgs) == rows * cols

            w, h = imgs[0].size
            grid = Image.new("RGB", size=(cols * w, rows * h))
            grid_w, grid_h = grid.size

            for i, img in enumerate(imgs):
                grid.paste(img, box=(i % cols * w, i // cols * h))
            return grid

        with open(args.inference_prompt_file, "r") as f:
            prompt_list = list(f.read().split(args.inference_prompt_split_token))

        all_images = []
        pipeline = pipeline.to("cuda")

        if text_encoder_name == args.pretrained_model_name_or_path:
            pipeline.text_encoder.__class__.forward = (
                pipeline.text_encoder.__class__.forward_original
            )

        for prompt in prompt_list:
            with autocast("cuda"):
                generator = torch.Generator("cuda").manual_seed(10)
                images = pipeline(
                    [prompt] * args.inference_prompt_number_per_prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    generator=generator,
                ).images
                all_images.extend(images)

        grid = image_grid(
            all_images, len(prompt_list), args.inference_prompt_number_per_prompt
        )

        if args.inference_prompt_output_file is not None:
            grid.save(args.inference_prompt_output_file)
        else:
            grid.save(
                "output_"
                + str(args.learning_rate)
                + "_"
                + str(args.max_train_steps)
                + "_"
                + str(args.train_batch_size)
                + "_"
                + str(random.randint(0, 9999999999))
                + ".jpg"
            )
