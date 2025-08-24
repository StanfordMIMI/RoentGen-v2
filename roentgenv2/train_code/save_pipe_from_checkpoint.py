### Some sections of this code used the code of the huggingface diffusers repository.
### In particular from the file https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
### And the file https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

### Imports ###
import itertools
import logging
import math
import os, sys
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

# if is_wandb_available():
#     import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

from dataset import UnetFinetuningDataset
from dataset_wds import RGFineTuningWebDataset, RGFineTuningWebDatasetMetadata
from train_loop import train_loop
from pipeline import create_and_save_pipeline
from models import load_models, EMAModel


##########################################################
def main(args):

    ### Preliminaries ###
    logging_dir = Path(args.output_dir, args.logging_dir)
    logger = get_logger(__name__)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        split_batches=True,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # if args.report_to == "wandb":
    #     if not is_wandb_available():
    #         raise ImportError(
    #             "Make sure to install wandb if you want to use it for logging during training."
    #         )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # TODO: check if i can remove this
    if accelerator.is_main_process:
        if args.do_not_save_weights:
            logger.info(
                "Notice that weights will not be saved at the end of the execution"
            )
        else:
            logger.info("Weights will be saved at the end")

        logger.info("accelerator.num_processes {}".format(accelerator.num_processes))

    # Handle the output directory creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        else:
            raise Exception("Nowhere to store the weights! please specify output_dir")

    # Save the configuration in the output directory
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as file:
        yaml.dump(args.get_config(), file)

    ##########################################################
    ### Get the models: text_encoder, vae, unet, tokenizer ###
    (
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
    ) = load_models(args, accelerator, logger)

    # Freeze vae and text_encoder
    if args.image_type == "pt":
        vae.requires_grad_(False)

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.train_text_encoder and freeze_pooler:
        text_encoder.pooler.requires_grad_(False)
    ##########################################################
    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("succeeded in enabling xformers")
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
    else:
        logger.info("xformer not available")

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Scale the learning rate by the number of processes and the gradient accumulation steps
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        if accelerator.is_main_process:
            logger.info("using adam 8-bit")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        if accelerator.is_main_process:
            logger.info("NOT using adam 8-bit")
        optimizer_class = torch.optim.AdamW

    ### Prepare the optimizer ###
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters())
        if args.train_text_encoder
        else unet.parameters()
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    ##########################################################
    ### Prepare the dataset and the dataloader ###
    if accelerator.is_main_process:
        print(
            f"data_filter_file: {args.data_filter_file}",
        )
        print(f"loss_weights_file: {args.loss_weights_file}")

    if args.use_wds_dataset:
        logger.info(f"Using Webdataset at {args.url_root}")
        url_list = [
            os.path.join(args.url_root, x)
            for x in os.listdir(args.url_root)
            if x.endswith(".tar")
        ]

        # USER: choose if using the report with metadata or without
        train_dataset = RGFineTuningWebDataset(
            url_list=url_list,
            tokenizer=tokenizer,
            data_filter_file=args.data_filter_file,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
        )

        logger.info(f"dataset len {len(train_dataset)}")
        logger.info(f"dataloader len {len(train_dataloader)}")
    else:
        train_dataset = UnetFinetuningDataset(
            image_dir=args.image_dir,
            prompt_dir=args.prompt_dir,
            data_filter_file=args.data_filter_file,
            data_filter_split_token=args.data_filter_split_token,
            loss_weights_file=args.loss_weights_file,
            loss_weights_split_token=args.loss_weights_split_token,
            tokenizer=tokenizer,
            size=args.resolution,
            center_crop=args.center_crop,
            image_type=args.image_type,
        )

        with accelerator.main_process_first():
            if args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(args.max_train_samples))

        if accelerator.is_main_process:
            print("len(train_dataset)", len(train_dataset))

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=train_dataset.collate_fn_training,
            batch_size=args.train_batch_size,
        )
    ##########################################################
    # LR Scheduler and math around the number of training steps.
    logger.info(f"dataloader len {len(train_dataloader)}")
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        logger.info("lr_scheduler {}".format(lr_scheduler))

    ##########################################################
    ### Prepare everything with accelerator ###
    if args.train_text_encoder:
        (
            unet,
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    assert args.mixed_precision == accelerator.mixed_precision
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.image_type == "pt":
        vae.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        if args.train_text_encoder:
            logger.info("training text encoder")
        if not args.train_text_encoder:
            logger.info("NOT training the text encoder")

    # Create EMA for the unet.
    if args.use_ema:
        logger.info("USING EMA")
        ema_unet = EMAModel(unet.parameters())
    else:
        ema_unet = None

    logger.info(f"dataloader len {len(train_dataloader)}")
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("unet-fine-tuning", config=vars(args))

    ##########################################################
    ### Train! ###
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
            first_epoch = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(os.path.join(args.output_dir, path))
            accelerator.load_state(checkpoint_path)
            global_step = 0  # int(path.split("-")[1])
            accelerator.print(
                f"Loaded state from checkpoint. Starting at step {global_step}"
            )

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    ### Train loop ###
    # (
    #     logger,
    #     args,
    #     accelerator,
    #     train_dataloader,
    #     unet,
    #     text_encoder,
    #     vae,
    #     noise_scheduler,
    #     weight_dtype,
    #     optimizer,
    #     lr_scheduler,
    #     ema_unet,
    #     progress_bar,
    # ) = train_loop(
    #     logger,
    #     args,
    #     initial_global_step,
    #     first_epoch,
    #     accelerator,
    #     train_dataloader,
    #     unet,
    #     text_encoder,
    #     vae,
    #     noise_scheduler,
    #     weight_dtype,
    #     optimizer,
    #     lr_scheduler,
    #     ema_unet,
    # )

    accelerator.wait_for_everyone()

    ##########################################################
    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        create_and_save_pipeline(
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
        )
    ##########################################################

    accelerator.end_training()


from config import get_args_from_config

if __name__ == "__main__":
    print(f"Launching unet fine-tuning, pid:{os.getpid()}")

    # Pass args directly from command line
    # args = parse_args_train()

    # Pass args from config.yaml
    args = get_args_from_config()

    checkpoint_root = (
        "/p/scratch/ccstdl/moroianu1/checkpoints/roentgen2/0709_mimic_PA_race_gpt/"
    )
    for ck in os.listdir(checkpoint_root):
        if ck in [
            "checkpoint-22500",
            "checkpoint-12500",
            "checkpoint-7500",
        ]:
            args.resume_from_checkpoint = os.path.join(checkpoint_root, ck)
            args.output_dir = os.path.join(args.resume_from_checkpoint, "pipeline")

            main(args)

# python /p/project/ccstdl/moroianu1/roentgen_v2/rg2/train_code/save_pipe_from_checkpoint.py --config_file=/p/project/ccstdl/moroianu1/roentgen_v2/rg2/scripts/train_config.yaml
