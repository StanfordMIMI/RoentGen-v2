import os
import inspect
import shutil

import torch
import torch.utils.checkpoint
from tqdm.auto import tqdm

from accelerate import Accelerator, PartialState
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logger = get_logger(__name__)

from diffusers.utils.import_utils import is_xformers_available
from diffusers import PNDMScheduler

from dataset_inference import UnetInferenceDatasetCSV
from utils import numpy_to_pil
from models import load_models
from inference_config import get_args_from_config


################################################################
def main(args):
    # Create the accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        split_batches=True,
    )

    if accelerator.is_main_process:
        print(f"model_name_or_path: {args.model_name_or_path}")
        print(f"num_images_per_prompt: {args.num_images_per_prompt}")
        print(f"output_dir: {args.output_dir}")
        print(f"prompt_dir: {args.prompt_dir}")
        print(f"data_filter_file: {args.data_filter_file}")
        print(f"guidance_scale: {args.guidance_scale}")
        print(f"num_inference_steps: {args.num_inference_steps}")
        print(f"mixed_precision: {args.mixed_precision}")
        print(f"resume_from_latest: {args.resume_from_latest}")

    print(
        f"*** Using {accelerator.device} device, num_processes={accelerator.num_processes}, torch.cuda.device_count()={torch.cuda.device_count()}"
    )
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Set the seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output directory creation. This will remove any previous files at that path!
    assert args.output_dir[-1] == "/"
    if accelerator.is_main_process:
        if args.output_dir is not None:
            if os.path.isdir(args.output_dir) and not args.resume_from_latest:
                shutil.rmtree(args.output_dir)
            if os.path.isfile(args.output_dir[:-1] + ".zip"):
                os.remove(args.output_dir[:-1] + ".zip")
            os.makedirs(args.output_dir, exist_ok=True)
            print("Savefolder ready.")
        else:
            raise Exception("nowhere to store the images")

    ################################################################
    # Load the trained models from the paths/names specified in args
    text_encoder, tokenizer, vae, unet = load_models(args)
    print(f"Models created, pid:{os.getpid()}")

    # Enable xformers if available
    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("Succeeded in enabling xformers.")
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
    else:
        print("xformers not available")

    # Freeze the models
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create the noise scheduler
    noise_scheduler = PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        set_alpha_to_one=False,
        skip_prk_steps=True,
        steps_offset=1,
        trained_betas=None,
    )

    ################################################################
    # Create the dataset and dataloader
    print("Using csv inference prompt csv at ", args.csv_file_prompts)
    test_dataset = UnetInferenceDatasetCSV(
        csv_file=args.csv_file_prompts, tokenizer=tokenizer
    )

    print("Dataset size:", len(test_dataset))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=test_dataset.collate_fn_inference,
        batch_size=args.test_batch_size,
    )

    print(
        "Dataloader size:", len(test_dataloader), ", batch size:", args.test_batch_size
    )

    ################################################################
    # Prepare everything with Accelerate
    # To avoid issues with multi-gpu accelerate, we will put only the numerical part of batch on device separately,
    # since we also have text inputs in the batch
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    distributed_state = PartialState()

    accelerator.wait_for_everyone()

    ################################################################
    # If inference job got interrupted, resume at latest batch index
    if args.resume_from_latest:
        if os.path.isfile(os.path.join(args.output_dir, "latest_batch_index.txt")):
            with open(
                os.path.join(args.output_dir, "latest_batch_index.txt"), "r"
            ) as f:
                resume_index = int(f.read())
            print("Resuming inference at batch index", resume_index)
        else:
            print("No latest_batch_index.txt file found, starting from first batch")
            resume_index = 0

    ################################################################
    # Inference loop
    height = args.resolution
    width = args.resolution

    progress_bar = tqdm(
        range(0, len(test_dataloader)),
        initial=resume_index,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Batch")

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            if step < resume_index:
                pass
            else:
                print(f"*** Processing batch {step}")
                print(f"*** Batch size: {len(batch['input_ids'])}")
                with distributed_state.split_between_processes(
                    batch["input_ids"]
                ) as input_ids:
                    with distributed_state.split_between_processes(
                        batch["uncond_ids"]
                    ) as uncond_ids:
                        with distributed_state.split_between_processes(
                            batch["prompt_texts"]
                        ) as prompt:
                            with distributed_state.split_between_processes(
                                batch["stem_values"]
                            ) as stem:

                                input_ids = input_ids.to(accelerator.device)
                                uncond_ids = uncond_ids.to(accelerator.device)
                                # Conditional text embeddings
                                text_embeddings = text_encoder(input_ids=input_ids)[0]
                                bs_embed, seq_len, _ = text_embeddings.shape
                                text_embeddings = text_embeddings.repeat(
                                    1, args.num_images_per_prompt, 1
                                )
                                text_embeddings = text_embeddings.view(
                                    bs_embed * args.num_images_per_prompt, seq_len, -1
                                )

                                # Unconditional text embeddings
                                uncond_embeddings = text_encoder(input_ids=uncond_ids)[
                                    0
                                ]
                                seq_len = uncond_embeddings.shape[1]
                                uncond_embeddings = uncond_embeddings.repeat(
                                    1, args.num_images_per_prompt, 1
                                )
                                uncond_embeddings = uncond_embeddings.view(
                                    bs_embed * args.num_images_per_prompt, seq_len, -1
                                )

                                # Concatenate the unconditional and conditional text embeddings
                                text_embeddings = torch.cat(
                                    [uncond_embeddings, text_embeddings]
                                )

                                # Latents
                                latents_shape = (
                                    input_ids.shape[0] * args.num_images_per_prompt,
                                    unet.config.in_channels,
                                    height // 8,
                                    width // 8,
                                )

                                assert input_ids.shape[0] == bs_embed

                                generator = torch.Generator(
                                    accelerator.device
                                ).manual_seed(step)

                                latents_dtype = text_embeddings.dtype

                                # Sample latents from the normal distribution
                                latents = torch.randn(
                                    latents_shape,
                                    generator=generator,
                                    device=accelerator.device,
                                    dtype=latents_dtype,
                                )

                                noise_scheduler.set_timesteps(args.num_inference_steps)
                                timesteps_tensor = noise_scheduler.timesteps.to(
                                    accelerator.device
                                )
                                latents = latents * noise_scheduler.init_noise_sigma

                                accepts_eta = "eta" in set(
                                    inspect.signature(
                                        noise_scheduler.step
                                    ).parameters.keys()
                                )
                                extra_step_kwargs = {}
                                if accepts_eta:
                                    extra_step_kwargs["eta"] = 0.0

                                accepts_generator = "generator" in set(
                                    inspect.signature(
                                        noise_scheduler.step
                                    ).parameters.keys()
                                )
                                if accepts_generator:
                                    extra_step_kwargs["generator"] = generator

                                for i, t in enumerate(timesteps_tensor):
                                    # expand the latents if we are doing classifier free guidance
                                    latent_model_input = torch.cat([latents] * 2)
                                    latent_model_input = (
                                        noise_scheduler.scale_model_input(
                                            latent_model_input, t
                                        )
                                    )

                                    # predict the noise residual
                                    noise_pred = unet(
                                        latent_model_input,
                                        t,
                                        encoder_hidden_states=text_embeddings,
                                    ).sample

                                    # perform guidance
                                    noise_pred_uncond, noise_pred_text = (
                                        noise_pred.chunk(2)
                                    )
                                    noise_pred = (
                                        noise_pred_uncond
                                        + args.guidance_scale
                                        * (noise_pred_text - noise_pred_uncond)
                                    )

                                    # compute the previous noisy sample x_t -> x_t-1
                                    latents = noise_scheduler.step(
                                        noise_pred, t, latents, **extra_step_kwargs
                                    ).prev_sample

                                # Legacy fudge factor from original dreambooth code
                                latents = 1 / 0.18215 * latents
                                images = vae.decode(latents).sample
                                images = (images / 2 + 0.5).clamp(0, 1)

                                images = (
                                    images.cpu().permute(0, 2, 3, 1).float().numpy()
                                )

                                # A list of images
                                images = numpy_to_pil(images)

                                # Save the generated images to disk
                                for i, stem_i in enumerate(stem):
                                    # Save each image in the batch as a jpg
                                    for j in range(args.num_images_per_prompt):
                                        images[i * args.num_images_per_prompt + j].save(
                                            args.output_dir
                                            + f"{stem_i}_{str(j)}.jpg"
                                        )
                                    # Save the prompt text
                                    with open(
                                        args.output_dir
                                        + f"{stem_i}.txt",
                                        "w",
                                    ) as f:
                                        f.write(prompt[i])

                                # To get the allocated memory
                                allocated_memory = torch.cuda.memory_allocated()
                                # To get the reserved memory (total memory that PyTorch has reserved from the GPU driver)
                                reserved_memory = torch.cuda.memory_reserved()

                                print(
                                    f"*** Allocated Memory: {allocated_memory / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
                                )
                                print(
                                    f"*** Reserved Memory: {reserved_memory / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
                                )

                torch.cuda.empty_cache()
                with open(
                    os.path.join(
                        args.output_dir,
                        f"latest_batch_index.txt",
                    ),
                    "w",
                ) as f:
                    f.write(str(step + 1))

                progress_bar.update(1)
                break  # for debugging, break after first batch
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Zip the generated images
        shutil.make_archive(args.output_dir[:-1], "zip", args.output_dir)
        print("Inference complete. Images zipped.")


if __name__ == "__main__":
    print("Launching inference")
    args = get_args_from_config()
    main(args)
