import itertools, os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import shutil


def train_loop(
    logger,
    args,
    initial_global_step,
    first_epoch,
    accelerator,
    train_dataloader,
    unet,
    text_encoder,
    vae,
    noise_scheduler,
    weight_dtype,
    optimizer,
    lr_scheduler,
    ema_unet,
):
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    global_step = initial_global_step
    unet.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info("Epoch {}, global step {}".format(epoch, global_step))

        if args.train_text_encoder:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            logger.info("*** batch {} ***".format(batch["pixel_values"].shape))
            with accelerator.accumulate(unet):
                # Convert images to latent space
                if args.image_type == "pt":
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    # Legacy fudge factor from original dreambooth code
                    latents = latents * 0.18215  # vae.config.scaling_factor
                elif args.image_type == "parameters":
                    latents = batch["pixel_values"].to(dtype=weight_dtype)
                else:
                    raise Exception("not supported")
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                text_input_ids = batch["input_ids"]
                if args.use_attention_mask:
                    attention_mask = batch["attention_mask"]
                else:
                    attention_mask = None

                prompt_embeds = text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=attention_mask,
                    return_dict=False,
                )
                encoder_hidden_states = prompt_embeds[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    print(
                        "are you sure ? --> noise_scheduler.config.prediction_type == v_prediction"
                    )
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Predict the noise residual
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Compute instance loss
                loss = F.mse_loss(
                    noise_pred.float(), target.float(), reduction="none"
                ).mean([1, 2, 3])
                loss_weights = batch["loss_weights"].to(dtype=weight_dtype)
                loss = (loss * loss_weights).sum() / loss_weights.sum()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                # Save state checkpoint
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # TODO: add validation pass

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    logger.info("Training finished")
    return (
        logger,
        args,
        accelerator,
        train_dataloader,
        unet,
        text_encoder,
        vae,
        noise_scheduler,
        weight_dtype,
        optimizer,
        lr_scheduler,
        ema_unet,
        progress_bar,
    )
