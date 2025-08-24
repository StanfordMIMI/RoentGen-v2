import os
import inspect
import shutil
import numpy as np
import time
import gc

import torch
import torch.utils.checkpoint
from tqdm.auto import tqdm

from accelerate import Accelerator, PartialState
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logger = get_logger(__name__)

from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers import PNDMScheduler

from dataset_inference import UnetInferenceDatasetCSV
from utils import numpy_to_pil
from models import load_models
from monai.transforms import ToTensor, ScaleIntensityd
import monai
import torchvision
import pandas as pd

if is_wandb_available():
    import wandb

from sex_model import get_sex_model_imperial
import torchxrayvision as xrv

from inference_config import get_args_from_config
import psutil

################################################################
class QCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        dicom_ids,
        demographics_csv,
    ):
        if images.shape[3] == 3:
            self.images = np.moveaxis(images, 3, 1)
        else:
            self.images = images
        self.dicom_ids = dicom_ids
        self.transform = torchvision.transforms.Compose(
            [
                monai.transforms.Resize((224, 224)),
                ToTensor(),
            ]
        )
        self.demographics_df = pd.read_csv(demographics_csv)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = {}

        sample["dicom_id"] = self.dicom_ids[idx]

        image = self.images[idx]
        image = self.transform(image)
        sample["jpg"] = image

        my_df = self.demographics_df[
            self.demographics_df["dicom_id"] == sample["dicom_id"]
        ]
        sample["gender"] = my_df["gender"].values[0]
        sample["age"] = my_df["anchor_age"].values[0]
        sample["race"] = my_df["race"].values[0]
        return sample


################################################################
def check_sex(qc, dataloader, accelerator, args):

    print(
        f"***Start check_sex(): Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )
    qc_sex = np.ones(len(qc))

    progress_bar = tqdm(
        range(0, len(dataloader)),
        initial=0,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description(f"Sex (gpu {torch.cuda.current_device()}):")

    sex_targets = []
    sex_preds = []

    sex_model = get_sex_model_imperial(args.sex_model_path).to(accelerator.device)
    sex_model.eval()

    scale_transform = ScaleIntensityd(keys=["jpg"], minv=0, maxv=255)
    # DBEUG: make sure that the index in the dataloader is correct wrt the image list/dicom list
    with torch.no_grad():
        for batch in dataloader:
            sex_targets.append(str(batch["gender"][0]))
            image = scale_transform(batch)["jpg"].to(accelerator.device)
            pred = torch.softmax(sex_model(image), dim=1)
            y_pred = pred.argmax().detach().cpu().numpy()
            sex_preds.append(y_pred)
            progress_bar.update(1)

    sex_preds = ["F" if x == 1 else "M" for x in sex_preds]
    assert len(qc) == len(sex_preds)
    for i in range(len(qc)):
        if sex_preds[i] != sex_targets[i]:
            qc[i] = 0
            qc_sex[i] = 0

    print(
        f"***Mid check_sex(): Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )

    # Clean up gpu memory
    for var_name, var_value in locals().items():
        if isinstance(var_value, torch.Tensor):
            # print(f"On gpu:{var_name}, {var_value.is_cuda}")
            var_value = var_value.cpu()
        elif isinstance(var_value, torch.nn.Module):
            # print(f"On gpu:{var_name}, {next(var_value.parameters()).device}")
            var_value = var_value.cpu()

    del sex_model
    del image
    del pred
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"***End check_sex: Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )
    return qc, qc_sex


################################################################
def check_race(qc, dataloader, accelerator):
    print(
        f"***Start check_race(): Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )
    qc_race = np.ones(len(qc))

    progress_bar = tqdm(
        range(0, len(dataloader)),
        initial=0,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description(f"Race (gpu {torch.cuda.current_device()}):")

    race_preds = []
    race_targets = []

    race_model = xrv.baseline_models.emory_hiti.RaceModel().to(accelerator.device)

    scale_transform = ScaleIntensityd(keys=["jpg"], minv=-1024, maxv=1024)

    with torch.no_grad():
        for batch in dataloader:
            image = scale_transform(batch)["jpg"].to(accelerator.device)
            image = image[:, 0, :, :].unsqueeze(1)
            pred = race_model(image)
            pred_prob = torch.nn.Softmax()(pred).detach().cpu().numpy()
            y_pred = race_model.targets[pred_prob.argmax()]
            race_preds.append(y_pred)

            race_label = str(batch["race"][0])
            race_targets.append(race_label.capitalize())
            progress_bar.update(1)

    assert len(qc) == len(race_preds)
    for i, target in enumerate(race_targets):
        if target in ["White", "Black", "Asian"]:
            if target != race_preds[i]:
                qc[i] = 0
                qc_race[i] = 0
    print(
        f"***Mid check_race: Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )

    # Clean up gpu memory
    for var_name, var_value in locals().items():
        if isinstance(var_value, torch.Tensor):
            print(f"On gpu:{var_name}, {var_value.is_cuda}")
            var_value = var_value.cpu()
        elif isinstance(var_value, torch.nn.Module):
            print(f"On gpu:{var_name}, {next(var_value.parameters()).device}")
            var_value = var_value.cpu()

    del race_model
    del image
    del pred
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"***End check_race(): Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )
    return qc, qc_race


################################################################
def check_age(qc, dataloader, accelerator, sigma=16.0):
    print(
        f"***Start check_age(): Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )
    qc_age = np.ones(len(qc))

    progress_bar = tqdm(
        range(0, len(dataloader)),
        initial=0,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description(f"Age (gpu {torch.cuda.current_device()}):")

    age_preds = []
    age_targets = []

    age_model = xrv.baseline_models.riken.AgeModel().to(accelerator.device)

    scale_transform = ScaleIntensityd(keys=["jpg"], minv=-1024, maxv=1024)

    with torch.no_grad():
        for batch in dataloader:
            age_targets.append(batch["age"].numpy()[0])
            image = scale_transform(batch)["jpg"].to(accelerator.device)
            pred = age_model(image[:, 0, :, :])
            age_pred = pred.detach().cpu().numpy()[0][0]
            age_preds.append(np.rint(age_pred))
            progress_bar.update(1)

    assert len(qc) == len(age_preds)
    for i in range(len(qc)):
        if np.abs(age_targets[i] - age_preds[i]) > sigma:
            qc[i] = 0
            qc_age[i] = 0

    print(
        f"***Mid check_age(): Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )

    # Clean up gpu memory
    for var_name, var_value in locals().items():
        if isinstance(var_value, torch.Tensor):
            print(f"On gpu:{var_name}, {var_value.is_cuda}")
            var_value = var_value.cpu()
        elif isinstance(var_value, torch.nn.Module):
            print(f"On gpu:{var_name}, {next(var_value.parameters()).device}")
            var_value = var_value.cpu()

    del age_model
    del image
    del pred
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"***End check_age(): Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )
    return qc, qc_age


################################################################
def quality_check(images, stem_values, accelerator, args):
    print(
        f"***Start quality_check(): Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )
    print(f"Using demographics csv at {args.csv_demographics_check}")
    qc = np.ones(len(images))
    dataset = QCDataset(
        images,
        dicom_ids=stem_values,
        demographics_csv=args.csv_demographics_check,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Running QC on gpu {torch.cuda.current_device()} for {len(images)} images")
    qc, qc_sex = check_sex(qc, dataloader, accelerator, args)
    qc, qc_race = check_race(qc, dataloader, accelerator)
    qc, qc_age = check_age(qc, dataloader, accelerator)

    # Clean up gpu memory
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"***End quality_check(): Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )
    return qc, qc_sex, qc_race, qc_age


def inference_loop(
    accelerator,
    text_encoder,
    vae,
    unet,
    noise_scheduler,
    input_ids,
    uncond_ids,
    prompt,
    stem,
    args,
    step,
):
    print(
        f"***Start inference_loop(): Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )
    with torch.no_grad():
        input_ids = input_ids.to(accelerator.device)
        uncond_ids = uncond_ids.to(accelerator.device)
        # Conditional text embeddings
        text_embeddings = text_encoder(input_ids=input_ids)[0]
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, args.num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * args.num_images_per_prompt, seq_len, -1
        )

        # Unconditional text embeddings
        uncond_embeddings = text_encoder(input_ids=uncond_ids)[0]
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(1, args.num_images_per_prompt, 1)
        uncond_embeddings = uncond_embeddings.view(
            bs_embed * args.num_images_per_prompt, seq_len, -1
        )

        # Concatenate the unconditional and conditional text embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Latents
        height = args.resolution
        width = args.resolution
        latents_shape = (
            input_ids.shape[0] * args.num_images_per_prompt,
            unet.config.in_channels,
            height // 8,
            width // 8,
        )

        assert input_ids.shape[0] == bs_embed

        # Using step to change the seed
        generator = torch.Generator(accelerator.device).manual_seed(step)

        latents_dtype = text_embeddings.dtype

        # Sample latents from the normal distribution
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=accelerator.device,
            dtype=latents_dtype,
        )

        noise_scheduler.set_timesteps(args.num_inference_steps)
        timesteps_tensor = noise_scheduler.timesteps.to(accelerator.device)
        latents = latents * noise_scheduler.init_noise_sigma

        accepts_eta = "eta" in set(
            inspect.signature(noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = 0.0

        accepts_generator = "generator" in set(
            inspect.signature(noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        for i, t in enumerate(timesteps_tensor):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = noise_scheduler.scale_model_input(
                latent_model_input, t
            )

            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        # Legacy fudge factor from original dreambooth code
        latents = 1 / 0.18215 * latents
        images = vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)

    images = images.detach().cpu().permute(0, 2, 3, 1).float().numpy()

    print(
        f"***Mid inference_loop(): Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )

    # Clean up gpu memory
    for var_name, var_value in locals().items():
        if isinstance(var_value, torch.Tensor):
            print(f"On gpu:{var_name}, {var_value.is_cuda}")
            var_value = var_value.cpu()

    del input_ids
    del uncond_ids
    del text_embeddings
    del uncond_embeddings
    del latents
    del timesteps_tensor
    del t
    del latent_model_input
    del noise_pred
    del noise_pred_uncond
    del noise_pred_text
    del generator
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"***End inference_loop() after clean: Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )
    return images


def save_images_to_disk(images, prompt, stem, args, accelerator):
    print(
        f"***Start save_images_to_disk(): Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
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
            args.output_dir + f"{stem_i}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(prompt[i])
    print(
        f"***End save_images_to_disk(): Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )


################################################################
def main(args):
    # Initialize a wandb run in offline mode
    run = wandb.init(mode="offline")

    # Check if the run initialized successfully
    if run:
        print("WandB logs are saved in:", run.dir)
    else:
        print("Failed to initialize WandB run.")
    # Create the accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        split_batches=True,
        log_with=args.report_to,
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
        f"***Using {accelerator.device} device, num_processes={accelerator.num_processes}, torch.cuda.device_count()={torch.cuda.device_count()}"
    )
    print(
        f"***Code start: Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
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
    # test_dataloader = accelerator.prepare(test_dataloader)

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    distributed_state = PartialState()

    if accelerator.is_main_process:
        accelerator.init_trackers("QC_inference", config=vars(args))

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
    print(
        f"***Before inference loop: Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
    )

    progress_bar = tqdm(
        range(0, len(test_dataloader)),
        initial=resume_index,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Batch")

    # Prepare the failed QC info file
    failed_csv = os.path.join(
        args.output_dir,
        f"failed_QC_dicoms.csv",
    )
    if not os.path.isfile(failed_csv):
        with open(
            failed_csv,
            "w",
        ) as f:
            f.write("fail_QC_dicom\tqc_sex\tqc_race\tqc_age\n")

    start_time_zero = time.time()
    for step, batch in enumerate(test_dataloader):
        # Batch start time
        start_time = time.time()

        # If inference job got interrupted, resume at latest batch index
        if step < resume_index:
            pass
        else:
            print(f"*** Processing batch {step}")
            local_batch_size = len(batch["input_ids"])
            print(f"*** Batch size: {local_batch_size}")
            # Split the batch elements across gpu devices for distributed inference
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
                            with torch.no_grad():
                                print(
                                    f"***Batch start: Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
                                )
                                # Run inference for image generation
                                input_ids = input_ids.squeeze()
                                uncond_ids = uncond_ids.squeeze()
                                print(input_ids.shape, uncond_ids.shape)
                                images = inference_loop(
                                    accelerator,
                                    text_encoder,
                                    vae,
                                    unet,
                                    noise_scheduler,
                                    input_ids,
                                    uncond_ids,
                                    prompt,
                                    stem,
                                    args,
                                    step,
                                )

                                qc_start_time = time.time()
                                # Note! This code will error if more than 1 image generated per prompt
                                # because of the way QC dataset is constructed from images and dicom_ids
                                try:
                                    assert len(images) == len(stem)
                                except AssertionError:
                                    print(
                                        "Check inference config and set num_images_per_prompt to 1 for this script"
                                    )
                                # Quality check
                                # Obtain a list of indexes 0 or 1
                                # 0: failed QC, needs to be regenerated
                                # 1: passed QC, can be saved
                                print(
                                    f"QC start: Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
                                )
                                quality_index, qc_sex, qc_race, qc_age = quality_check(
                                    images, stem, accelerator, args
                                )

                                # Clean up gpu memory
                                gc.collect()
                                torch.cuda.empty_cache()
                                print(
                                    f"QC end: Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
                                )

                                # If all images passed QC, save the entire batch to disk
                                if np.sum(quality_index) == len(quality_index):
                                    print(
                                        f"All images passed QC on gpu {torch.cuda.current_device()}"
                                    )
                                    save_images_to_disk(
                                        images,
                                        prompt,
                                        stem,
                                        args,
                                        accelerator,
                                    )
                                # Else, re-run inference for failed images
                                else:
                                    # Split the images into those that passed and those that failed QC
                                    images_qc_pass = np.array(
                                        [
                                            images[i]
                                            for i in range(len(images))
                                            if quality_index[i] == 1
                                        ]
                                    )

                                    # Get the stem (dicom_id) and prompt lists for the images that passed QC
                                    stem_qc_pass = [
                                        stem[i]
                                        for i in range(len(images))
                                        if quality_index[i] == 1
                                    ]
                                    prompt_qc_pass = [
                                        prompt[i]
                                        for i in range(len(images))
                                        if quality_index[i] == 1
                                    ]

                                    # Save the images that passed QC
                                    print(
                                        f"Saving {int(np.sum(quality_index))} images that passed QC... on gpu {torch.cuda.current_device()}"
                                    )
                                    save_images_to_disk(
                                        images_qc_pass,
                                        prompt_qc_pass,
                                        stem_qc_pass,
                                        args,
                                        accelerator,
                                    )

                                    # End of QC step: images that passed were saved, images that failed will be noted
                                    # For the images that failes, save the dicom ids and failed mode to csv
                                    stem_qc_fail = [
                                        stem[i]
                                        for i in range(len(images))
                                        if quality_index[i] == 0
                                    ]
                                    qc_sex_fail = [
                                        qc_sex[i]
                                        for i in range(len(images))
                                        if quality_index[i] == 0
                                    ]
                                    qc_race_fail = [
                                        qc_race[i]
                                        for i in range(len(images))
                                        if quality_index[i] == 0
                                    ]
                                    qc_age_fail = [
                                        qc_age[i]
                                        for i in range(len(images))
                                        if quality_index[i] == 0
                                    ]

                                    if len(stem_qc_fail) > 0:
                                        with open(
                                            os.path.join(
                                                args.output_dir,
                                                f"failed_QC_dicoms.csv",
                                            ),
                                            "a",
                                        ) as f:
                                            for stem, sex, race, age in zip(
                                                stem_qc_fail,
                                                qc_sex_fail,
                                                qc_race_fail,
                                                qc_age_fail,
                                            ):
                                                f.write(
                                                    f"{stem}\t{sex}\t{race}\t{age}\n"
                                                )
                                # End of if/else for QC
                                print(
                                    f"End QC loop: Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
                                )
                                del images
                                gc.collect()
                                torch.cuda.empty_cache()
                                print(
                                    f"Batch end: Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
                                )

                                qc_elapsed_time = time.time() - qc_start_time
                                print(
                                    f"QC duration (incl re-generation) in {qc_elapsed_time:.2f} seconds ({accelerator.device})"
                                )
                # end of this batch
                print(
                    f"Batch {step} processed in {time.time() - start_time:.2f} seconds ({accelerator.device})"
                )
                print(
                    f"-----> Roughly {(step+1) * local_batch_size * accelerator.num_processes} images in {time.time() - start_time_zero:.2f} seconds across all devices since start"
                )

                with open(
                    os.path.join(
                        args.output_dir,
                        f"latest_batch_index.txt",
                    ),
                    "w",
                ) as f:
                    f.write(str(step + 1))

                progress_bar.update(1)

                # DEBUG: Check which tensors and models are on gpu
                # for var_name, var_value in locals().items():
                #     if isinstance(var_value, torch.Tensor):
                #         print(f"Batch end On gpu:{var_name}, {var_value.is_cuda}")
                #         # var_value = var_value.cpu()
                #     elif isinstance(var_value, torch.nn.Module):
                #         print(
                #             f"Batch end On gpu:{var_name}, {next(var_value.parameters()).device}"
                #         )
                #         # var_value = var_value.cpu()
                gc.collect()
                torch.cuda.empty_cache()
                print(
                    f"***Batch end: Allocated/Reserved Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}/{torch.cuda.memory_reserved() / 1024**2:.2f} MB on device {(torch.cuda.current_device())} {accelerator.device}"
                )

        break # for debugging, break after first batch
        accelerator.wait_for_everyone()

    # if accelerator.is_main_process:
    #     # Zip the generated images
    #     shutil.make_archive(args.output_dir[:-1], "zip", args.output_dir)
    #     print("Inference complete. Images zipped.")



if __name__ == "__main__":
    global_start_time = time.time()
    print("Launching inference")
    args = get_args_from_config()

    # Get the overall system memory usage
    virtual_memory = psutil.virtual_memory()

    # Total physical memory in MB
    total_memory = virtual_memory.total / (1024**2)
    used_memory = virtual_memory.used / (1024**2)
    available_memory = virtual_memory.available / (1024**2)

    print(f"Total memory: {total_memory:.2f} MB")
    print(f"Used memory: {used_memory:.2f} MB")
    print(f"Available memory: {available_memory:.2f} MB")

    main(args)
    global_elapsed_time = time.time() - global_start_time
    print(
        f"*** Inference completed in {global_elapsed_time:.2f} seconds, find images at {args.output_dir}"
    )
