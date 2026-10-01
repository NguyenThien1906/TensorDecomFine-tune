import os
import gc
from typing import Callable, Any, Optional

import math
import yaml
import random
import secrets
import logging # for logging classes?
import itertools
from collections import defaultdict
import json
from pathlib import Path

from tqdm import tqdm
import wandb

import numpy as np

import torch
from torch.utils.data import DataLoader

import diffusers
from diffusers import (
    AutoencoderKL, EulerDiscreteScheduler, DDPMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline,
)
from diffusers.loaders import AttnProcsLayers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

import transformers
import transformers.utils.logging
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from .utils.registry import ClassRegistry
from .utils.model import get_layer_by_name
from .model.lora import (
    linear_layers, processors, LOrthogonalLoRACrossAttnProcessor,
    parameter_counter, ParameterCounter
)
from .model.tuckerlora import (
    TuckerLoRACrossAttnProcessor, TuckerLoRALinearLayer,
    TensorTrainCrossAttnProcessor, TensorTrainLinearLayer
)
from .model.pipeline_sdxl import StableDiffusionTLoRAPipeline
from .model.utils_sdxl import cast_training_params
from .data.dataset_sdxl import (
    ImageDataset, DreamBoothDataset, collate_fn, tokenize_prompt, encode_tokens, compute_time_ids,
    CsvDataset, collate_custom,
)

logger = get_logger(__name__)
trainers = ClassRegistry()

BASE_PROMPT = "a photo of {0}"

torch.backends.cuda.enable_flash_sdp(True)

@trainers.add_to_registry("sdxl_lora")
class LoraTrainerSDXL:

    def __init__(self, config):
        self.config = config

    def setup_exp_name(self, exp_idx):
        exp_name = "{0:0>3d}-{1}-{2}_{3}".format(
            exp_idx + 1,
            secrets.token_hex(2),
            f"{self.config.lora_rank}",
            f"{self.config.min_rank}"
        )
        return exp_name

    def setup_exp(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        # Get the last experiment idx - detect existing experiment by name
        exp_idx = 0
        for folder in os.listdir(self.config.output_dir):
            # noinspection PyBroadException
            try:
                curr_exp_idx = max(exp_idx, int(folder.split("-")[0].lstrip("0"))) #00000-
                exp_idx = max(exp_idx, curr_exp_idx)
            except:
                pass

        # Set up experiment name
        self.config.exp_name = self.setup_exp_name(exp_idx)

        # Set up experiment directory
        if not self.config.resume_training:
            self.config.output_dir = os.path.abspath(
                os.path.join(self.config.output_dir, self.config.exp_name)
            )

        # Check if experiment directory already exists, and avoid race condition
        if os.path.exists(self.config.output_dir) and not self.config.resume_training:
            raise ValueError(
                f"Experiment directory {self.config.output_dir} already exists. Race condition!"
            )

        # Create experiment directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Logs dir
        if not self.config.resume_training:
            self.config.logging_dir = os.path.join(self.config.output_dir, "logs")
            os.makedirs(self.config.logging_dir, exist_ok=True)

        # Save hyperparameters
        with open(os.path.join(self.config.logging_dir, "hparams.yml"), "w") as outfile:
            yaml.dump(vars(self.config), outfile)

    def setup_accelerator(self):
        # Set up wandb
        if self.config.wandb_api_key is not None:
            wandb.login(key=self.config.wandb_api_key)

        # Set up from accelerate lib with wandb tracking
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.config.output_dir
        )
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            log_with="wandb",
            project_config=accelerator_project_config,
        )
        if self.config.wandb_api_key is not None:
            self.accelerator.init_trackers(
                project_name=self.config.project_name,
                config=self.config,
                init_kwargs={
                    "wandb": {
                        "name": self.config.exp_name,
                        "settings": wandb.Settings(
                            code_dir=os.path.dirname(self.config.argv[1])
                        ),
                    }
                },
            )

        # Logging setup
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)

        # Set logging at only main process in order to have only one consistent logging thread.
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # Quantization
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

    def setup_base_model(self):
        self.scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="scheduler",
            revision=self.config.revision,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.config.revision,
        )
        # Core text encoder 
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.config.revision,
        )
        # Text encoder + linear projection from text embeddings to another embedding space
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=self.config.revision,
        )
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.config.revision,
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=self.config.revision,
        )

        # Gradient checkpointing
        #self.unet.enable_gradient_checkpointing()
        #self.unet.enable_xformers_memory_efficient_attention()

    def setup_model(self):
        # Training SD: freeze vae and text_encoder, train the UNet
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        # Set up LoRA training type (between TLoRA and LoRA) using ClassRegistry()
        attn_processor = processors[self.config.trainer_type]
        linear_layer = linear_layers[self.config.trainer_type]

        self.params_to_optimize = []
        attn_procs = {}
        # Apply LoRA to all attention blocks
        for name in self.unet.attn_processors.keys():
            # get attention dimension
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            # get latent dimension in attention blocks
            if name.startswith("mid_block"): # difference between layers
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            # configurations
            if cross_attention_dim:
                rank = min(cross_attention_dim, hidden_size, self.config.lora_rank)
            else:
                rank = min(hidden_size, self.config.lora_rank)
            
            kwargs = {
                "hidden_size": hidden_size,
                "cross_attention_dim": cross_attention_dim,
                "rank": rank,
                "linear_layer": linear_layer,
                "sig_type": self.config.sig_type,
                "do_training": True,
            }
            
            # flag original layer to note the processor belongs to
            if isinstance(attn_processor, LOrthogonalLoRACrossAttnProcessor):
                # get_layer_by_name: retrieve layer name inside UNet
                kwargs["original_layer"] = get_layer_by_name(self.unet, name.split(".processor")[0])

            # real setup for lora to each attention layer
            attn_procs[name] = attn_processor(**kwargs)

        # Set the attention processors
        self.unet.set_attn_processor(attn_procs) # re-apply newly constructed processors back to attention blocks
        
        # registration with accelerator library
        self.adaptive_layers = AttnProcsLayers(self.unet.attn_processors)
        self.accelerator.register_for_checkpointing(self.adaptive_layers)

        # Set trainable parameters and optimizer
        for name, param in self.adaptive_layers.named_parameters():
            if param.requires_grad == True:
                self.params_to_optimize.append(param) # registration to optimizer
        self.adaptive_layers.train()

        # Prepare the model, optimizer, and dataloader with the accelerator
        device = self.accelerator.device
        print(f"Moving models to device: {device}")
        self.unet.to(device)
        self.vae.to(device)
        self.text_encoder.to(device)
        self.text_encoder_2.to(device)

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.params_to_optimize,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

    def setup_lr_scheduler(self):
        pass

    def setup_dataset(self):
        # DreamBooth-style training
        if self.config.with_prior_preservation:
            self.train_dataset = DreamBoothDataset(
                instance_data_root=self.config.train_data_dir,
                instance_prompt=BASE_PROMPT.format(
                    f"{self.config.placeholder_token} {self.config.class_name}"
                ),
                class_data_root=(
                    self.config.class_data_dir
                    if self.config.with_prior_preservation
                    else None
                ),
                class_prompt=BASE_PROMPT.format(self.config.class_name),
                tokenizers=(self.tokenizer, self.tokenizer_2),
                size=self.config.resolution,
            )
            # convert a 2-parameter function to 1-parameter, like this below
            #def collator(examples):
            #    return collate_fn(examples, self.config.with_prior_preservation)
            collator: Optional[Callable[[Any], dict[str, torch.Tensor]]] = (
                lambda examples: collate_fn(
                    examples, self.config.with_prior_preservation
                )
            )
        else: # CsvDataset
            self.train_set = CsvDataset(
                csv_file=self.config.train_data_csv,
                image_dir=self.config.image_dir,
                tokenizers=(self.tokenizer, self.tokenizer_2),
                resolution=self.config.resolution,
            )
            self.test_set = CsvDataset(
                csv_file=self.config.test_data_csv,
                image_dir=self.config.image_dir,
                tokenizers=(self.tokenizer, self.tokenizer_2),
                resolution=self.config.resolution,
            )

            collator: Optional[Callable[[Any], dict[str, torch.Tensor]]] = (
                lambda examples: collate_custom(examples)
            )
            print(f"Train dataset length: {len(self.train_set)}")
            print(f"Test dataset length: {len(self.test_set)}")

        self.train_dataloader = DataLoader(
            self.train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=self.config.dataloader_num_workers,
            generator=self.generator,
        )
        self.test_dataloader = DataLoader(
            self.test_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=self.config.dataloader_num_workers,
            generator=self.generator,
        )

    # noinspection PyTypeChecker
    def move_to_device(self):
        # prepare accelerator for training, move unet and lora layers to accelerator device, cast to weight_dtype if needed
        self.adaptive_layers, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.adaptive_layers, self.optimizer, self.train_dataloader
        )

        # The base model weights are not trained, but they should still be moved to accelerator device and cast to weight_dtype if needed for mixed precision training
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype) # 83 M
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype) # 2.6 B
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype) # 817 M total
        self.text_encoder_2.to(self.accelerator.device, dtype=self.weight_dtype)

        # All trained parameters should be explicitly moved to float32 even for mixed precision training
        cast_training_params(
            (self.unet, self.text_encoder, self.text_encoder_2), dtype=torch.float32
        )

    def setup_seed(self):
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.generator = torch.Generator()
        self.generator.manual_seed(self.config.seed)

    def setup(self):
        self.setup_exp()
        self.setup_accelerator()
        self.setup_seed()

        self.setup_base_model()
        self.setup_model()
        self.setup_optimizer()
        self.setup_lr_scheduler()
        
        self.setup_dataset()
        
        self.move_to_device()
        self.setup_pipeline()

    def train_step(self, batch):
        # encode image to latents
        if self.config.with_prior_preservation: #DreamBooth
            latents = self.vae.encode(
                batch["pixel_values"].to(device=self.accelerator.device, dtype=self.weight_dtype)
            ).latent_dist.sample()
        else: #CsvDataset
            latents = self.vae.encode(
                batch["image"].to(device=self.accelerator.device, dtype=self.weight_dtype)
            ).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # pure noise & timestep
        noise = torch.randn_like(latents).to(device=self.accelerator.device)
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (latents.shape[0],),
            device=self.accelerator.device,
        )

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler.config.prediction_type}"
            )

        target = target.to(device=self.accelerator.device)

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps).to(device=self.accelerator.device)

        # Get encoder_hidden_states
        if not self.config.with_prior_preservation: # CsvDataset
            encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
                (self.text_encoder, self.text_encoder_2),
                (batch["input_ids"], batch["input_ids_2"]),
            )
        else: # DreamBooth
            encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
                (self.text_encoder, self.text_encoder_2),
                (batch["input_ids"], batch["input_ids_2"]),
            )
        encoder_hidden_states = encoder_hidden_states.to(device=self.accelerator.device)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(device=self.accelerator.device)

        add_time_ids = compute_time_ids(
            original_size=batch["original_sizes"],
            crops_coords_top_left=batch["crop_top_lefts"],
            resolution=self.config.resolution,
        ).to(device=self.accelerator.device)
        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": pooled_encoder_hidden_states,
        }
        outputs = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=unet_added_conditions,
        ).sample

        if self.config.with_prior_preservation:
            outputs, prior_outputs = torch.chunk(outputs, 2, dim=0)
            target, prior_target = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = torch.nn.functional.mse_loss(
                outputs.float(), target.float(), reduction="mean"
            )

            # Compute prior loss
            prior_loss = torch.nn.functional.mse_loss(
                prior_outputs.float(), prior_target.float(), reduction="mean"
            )

            # Add the prior loss to the instance loss.
            loss = loss + self.config.prior_loss_weight * prior_loss
        else:
            loss = torch.nn.functional.mse_loss(
                outputs.float(), target.float(), reduction="mean"
            )

        # delete variables to free up memory
        if self.config.with_prior_preservation:
            del prior_outputs, prior_target
        del outputs, target
        gc.collect()
        torch.cuda.empty_cache()

        return loss

    def setup_pipeline(self):
        # sampling algorithm
        scheduler = EulerDiscreteScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="scheduler"
        )
        # set up pipeline for inference and validation
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            scheduler=scheduler,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            unet=self.accelerator.unwrap_model(self.unet, keep_fp32_wrapper=False),
            vae=self.vae,
            revision=self.config.revision,
            torch_dtype=(
                self.weight_dtype
                if self.accelerator.mixed_precision in ["fp16", "bf16"]
                else torch.float16
            ),
        )

        self.pipeline.safety_checker = None
        self.pipeline = self.pipeline.to(self.accelerator.device)
        self.pipeline.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def validation(self, epoch):
        generator = torch.Generator(device=self.accelerator.device).manual_seed(42)
        prompts = self.config.validation_prompts.split('#')

        samples_path = os.path.join(
            self.config.output_dir,
            f"checkpoint-{epoch}",
            "samples",
            "validation"
        )
        os.makedirs(samples_path, exist_ok=True)

        all_images, all_captions = [], []
        for prompt in prompts:
            with torch.autocast("cuda"):
                caption = prompt.format(
                    f"{self.config.placeholder_token} {self.config.class_name}"
                )
                kwargs = {
                    "num_inference_steps": 25,
                    "guidance_scale": 5.0,
                    "prompt": caption,
                    "num_images_per_prompt": self.config.num_val_imgs_per_prompt,
                }
                images = self.pipeline(generator=generator, **kwargs).images
                gc.collect()
                torch.cuda.empty_cache()

            all_images += images
            all_captions += [caption] * len(images)

            os.makedirs(os.path.join(samples_path, caption), exist_ok=True)
            for idx, image in enumerate(images):
                image.save(os.path.join(samples_path, caption, f"{idx}.png"))

        for tracker in self.accelerator.trackers:
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=caption)
                        for image, caption in zip(all_images, all_captions)
                    ]
                }
            )
        torch.cuda.empty_cache()

    def save_model(self, epoch):
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{epoch}")
        os.makedirs(save_path, exist_ok=True)
        self.unet.save_attn_procs(os.path.join(save_path))

    def train(self):
        global parameter_counter
        print(f"Total parameters: {parameter_counter.get_count()}")

        # early stopping variables
        best_val_loss = math.inf
        epochs_no_improve = 0
        break_flag = False

        # get starting epoch
        start_epoch = 0
        if self.config.resume_training:
            if self.config.start_checkpoint is None:
                # get max checkpoint epoch by reading checkpoint folders "checkpoint-{}"
                config_path = Path(self.config.resume_config_path).parent.parent
                print(config_path)
                for folder in config_path.iterdir():
                    if folder.is_dir() and folder.name.startswith("checkpoint-"):
                        try:
                            epoch_num = int(folder.name.split("checkpoint-")[1])
                            start_epoch = max(start_epoch, epoch_num)
                        except:
                            pass
            else:
                start_epoch = self.config.start_checkpoint
    
            print(f"Resuming training from epoch {start_epoch}...")

        # tqdm is just to verbose training progress
        for epoch in tqdm(range(start_epoch, self.config.num_train_epochs), 
            initial=start_epoch, total=self.config.num_train_epochs):
            log_dict = {}
            
            #TRAINING
            avg_train_loss = 0.0
            for batch in self.train_dataloader:
                # step
                with self.accelerator.autocast():
                    loss = self.train_step(batch)

                self.accelerator.backward(loss)
                avg_train_loss += loss.item()

                # update & reset
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                del batch, loss
                gc.collect()
                torch.cuda.empty_cache()

            avg_train_loss /= len(self.train_dataloader)
            log_dict["avg_train_loss"] = avg_train_loss

            # VALIDATION
            avg_val_loss = 0.0
            with torch.no_grad():
                for batch in self.test_dataloader:
                    # step
                    with self.accelerator.autocast():
                        loss = self.train_step(batch)

                    avg_val_loss += loss.item()

                    # del
                    del batch, loss
                    gc.collect()
                    torch.cuda.empty_cache()

            avg_val_loss /= len(self.test_dataloader)
            log_dict["avg_val_loss"] = avg_val_loss

            # EARLY STOPPING CHECK
            if self.config.early_stopping_patience is not None:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    
                    # save best cp
                    if self.accelerator.is_main_process:
                        if self.config.wandb_api_key is not None and self.config.validation_prompts:
                            self.validation(epoch)
                        self.save_model(epoch)
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.config.early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch}."
                    )
                    break_flag = True

            # LOGGING AND TRACKING
            for tracker in self.accelerator.trackers:
                tracker.log(log_dict)

            # CHECKPOINT
            if self.accelerator.is_main_process:
                if epoch % self.config.checkpointing_steps == 0 and epoch >= start_epoch:
                    if self.config.wandb_api_key is not None and self.config.validation_prompts:
                        self.validation(epoch)
                    self.save_model(epoch)
            
            # early stopping
            if break_flag:
                print(f"Early stopping triggered at epoch {epoch}. Ending training.")
                break

        # FINAL
        end_epoch = self.config.num_train_epochs if not break_flag else epoch

        if self.accelerator.is_main_process:
            if self.config.wandb_api_key is not None and self.config.validation_prompts:
                self.validation(end_epoch)
            self.save_model(end_epoch)

        self.accelerator.end_training()

@trainers.add_to_registry("sdxl_tlora")
class TLoraTrainerSDXL(LoraTrainerSDXL):

    def __init__(self, config):
        super().__init__(config)

    def setup_exp_name(self, exp_idx):
        exp_name = "{0:0>5d}-{1}-{2}".format(
            exp_idx + 1,
            secrets.token_hex(2),
            os.path.basename(os.path.normpath(self.config.train_data_dir)),
        )
        exp_name += f"_t{self.config.trainer_type}{self.config.lora_rank}"
        return exp_name

    def get_mask_by_timestep(self, timestep, max_timestep, max_rank, min_rank=1, alpha=1):
        r = int(((max_timestep - timestep)/ max_timestep) ** alpha * (max_rank - min_rank)) + min_rank
        sigma_mask = torch.zeros((1, self.config.lora_rank))
        sigma_mask[:, :r] = 1.0
        return sigma_mask

    def train_step(self, batch):
        # encode image to latents
        if self.config.with_prior_preservation: #DreamBooth
            latents = self.vae.encode(
                batch["pixel_values"].to(self.weight_dtype)
            ).latent_dist.sample()
        else: #ImageDataset
            latents = self.vae.encode(
                batch["image"].to(self.weight_dtype) * 2.0 - 1.0
            ).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # pure noise & timestep
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        )

        # sigma mask in SVD based on time-step
        sigma_mask = self.get_mask_by_timestep(
            timestep=timesteps[0],
            max_timestep=self.scheduler.num_train_timesteps,
            max_rank=self.config.lora_rank,
            min_rank=self.config.min_rank,
            alpha=self.config.alpha_rank_scale,
        )

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler.config.prediction_type}"
            )

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get encoder_hidden_states
        if not self.config.with_prior_preservation: # ImageDataset
            input_ids_list = tokenize_prompt(
                (self.tokenizer, self.tokenizer_2),
                BASE_PROMPT.format(
                    f"{self.config.placeholder_token} {self.config.class_name}"
                ),
            )
            # prompt = BASE_PROMPT.format(
            #         f"{self.config.placeholder_token} {self.config.class_name}"
            #     )
            encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
                (self.text_encoder, self.text_encoder_2), input_ids_list
            )
        else: # DreamBooth
            encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
                (self.text_encoder, self.text_encoder_2),
                (batch["input_ids"], batch["input_ids_2"]),
            )
        add_time_ids = compute_time_ids(
            original_size=batch["original_sizes"],
            crops_coords_top_left=batch["crop_top_lefts"],
            resolution=self.config.resolution,
        )
        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": pooled_encoder_hidden_states,
        }
        outputs = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=unet_added_conditions,
            cross_attention_kwargs={
                "sigma_mask": sigma_mask.detach().to(encoder_hidden_states.device)
            },
        ).sample

        if self.config.with_prior_preservation:
            outputs, prior_outputs = torch.chunk(outputs, 2, dim=0)
            target, prior_target = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = torch.nn.functional.mse_loss(
                outputs.float(), target.float(), reduction="mean"
            )

            # Compute prior loss
            prior_loss = torch.nn.functional.mse_loss(
                prior_outputs.float(), prior_target.float(), reduction="mean"
            )

            # Add the prior loss to the instance loss.
            loss = loss + self.config.prior_loss_weight * prior_loss
        else:
            loss = torch.nn.functional.mse_loss(
                outputs.float(), target.float(), reduction="mean"
            )

        return loss

    def setup_pipeline(self):
        scheduler = EulerDiscreteScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.pipeline = StableDiffusionTLoRAPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            scheduler=scheduler,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            unet=self.accelerator.unwrap_model(self.unet, keep_fp32_wrapper=False),
            vae=self.vae,
            revision=self.config.revision,
            torch_dtype=(
                self.weight_dtype
                if self.accelerator.mixed_precision in ["fp16", "bf16"]
                else torch.float16
            ),
            max_rank=self.config.lora_rank,
            min_rank=self.config.min_rank,
            alpha=self.config.alpha_rank_scale,
        )

        self.pipeline.safety_checker = None
        self.pipeline = self.pipeline.to(self.accelerator.device)
        self.pipeline.set_progress_bar_config(disable=True)

@trainers.add_to_registry("sdxl_tucker_lora")
class TuckerLoRATrainerSDXL(LoraTrainerSDXL):

    def __init__(self, config):
        super().__init__(config)

    def setup_exp_name(self, exp_idx):
        exp_name = "{0:0>3d}-{1}-{2}-{3}".format(
            exp_idx + 1,
            secrets.token_hex(2),
            f"{self.config.trainer_type}",
            f"{self.config.tucker_mode}"
        )
        return exp_name

    def setup_pipeline(self):
        scheduler = EulerDiscreteScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            scheduler=scheduler,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            unet=self.accelerator.unwrap_model(self.unet, keep_fp32_wrapper=False),
            vae=self.vae,
            revision=self.config.revision,
            torch_dtype=(
                self.weight_dtype
                if self.accelerator.mixed_precision in ["fp16", "bf16"]
                else torch.float16
            )
        )

        self.pipeline.safety_checker = None
        self.pipeline = self.pipeline.to(self.accelerator.device)
        self.pipeline.set_progress_bar_config(disable=True)

    def setup_model(self):
        # Training SD: freeze vae and text_encoder, train the UNet
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        # Set up LoRA training type (between TLoRA and LoRA) using ClassRegistry()
        attn_processor = processors[self.config.trainer_type]
        linear_layer = linear_layers[self.config.trainer_type]

        self.params_to_optimize = []
        attn_procs = {}
        # Apply LoRA to all attention blocks
        for name in self.unet.attn_processors.keys():
            # get attention dimension
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            # get latent dimension in attention blocks
            if name.startswith("mid_block"): # difference between layers
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            # configurations
            kwargs = {
                "hidden_size": hidden_size,
                "mode": self.config.tucker_mode,
                "config_json": self.config.tucker_config_json,
                "cross_attention_dim": cross_attention_dim,
                "linear_layer": linear_layer,
            }
            
            # flag original layer to note the processor belongs to
            if isinstance(attn_processor, TuckerLoRACrossAttnProcessor):
                # get_layer_by_name: retrieve layer name inside UNet
                kwargs["original_layer"] = get_layer_by_name(self.unet, name.split(".processor")[0])

            # real setup for lora to each attention layer
            attn_procs[name] = attn_processor(**kwargs)

        # Set the attention processors
        self.unet.set_attn_processor(attn_procs) # re-apply newly constructed processors back to attention blocks
        
        # registration with accelerator library
        self.adaptive_layers = AttnProcsLayers(self.unet.attn_processors)
        self.accelerator.register_for_checkpointing(self.adaptive_layers)

        # Set trainable parameters and optimizer
        for name, param in self.adaptive_layers.named_parameters():
            if param.requires_grad == True:
                self.params_to_optimize.append(param) # registration to optimizer
        self.adaptive_layers.train() # set to train mode

        # Prepare the model, optimizer, and dataloader with the accelerator
        device = self.accelerator.device
        print(f"Moving models to device: {device}")
        self.unet.to(device)
        self.vae.to(device)
        self.text_encoder.to(device)
        self.text_encoder_2.to(device)

@trainers.add_to_registry("sdxl_tensor_train")
class TensorTrainTrainerSDXL(LoraTrainerSDXL):

    def __init__(self, config):
        super().__init__(config)

    def setup_exp_name(self, exp_idx):
        exp_name = "{0:0>5d}-{1}-{2}".format(
            exp_idx + 1,
            secrets.token_hex(2),
            os.path.basename(os.path.normpath(self.config.train_data_dir)),
        )
        exp_name += f"_t{self.config.trainer_type}{self.config.lora_rank}"
        return exp_name

    def train_step(self, batch):
        # encode image to latents
        if self.config.with_prior_preservation: #DreamBooth
            latents = self.vae.encode(
                batch["pixel_values"].to(self.weight_dtype)
            ).latent_dist.sample()
        else: #CsvDataset
            latents = self.vae.encode(
                batch["image"].to(self.weight_dtype) * 2.0 - 1.0
            ).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # pure noise & timestep
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        )

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler.config.prediction_type}"
            )

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get encoder_hidden_states
        if not self.config.with_prior_preservation: # CsvDataset
            encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
                (self.text_encoder, self.text_encoder_2),
                (batch["input_ids"], batch["input_ids_2"]),
            )
        else: # DreamBooth
            encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
                (self.text_encoder, self.text_encoder_2),
                (batch["input_ids"], batch["input_ids_2"]),
            )
        add_time_ids = compute_time_ids(
            original_size=batch["original_sizes"],
            crops_coords_top_left=batch["crop_top_lefts"],
            resolution=self.config.resolution,
        )
        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": pooled_encoder_hidden_states,
        }
        outputs = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=unet_added_conditions,
        ).sample

        if self.config.with_prior_preservation:
            outputs, prior_outputs = torch.chunk(outputs, 2, dim=0)
            target, prior_target = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = torch.nn.functional.mse_loss(
                outputs.float(), target.float(), reduction="mean"
            )

            # Compute prior loss
            prior_loss = torch.nn.functional.mse_loss(
                prior_outputs.float(), prior_target.float(), reduction="mean"
            )

            # Add the prior loss to the instance loss.
            loss = loss + self.config.prior_loss_weight * prior_loss
        else:
            loss = torch.nn.functional.mse_loss(
                outputs.float(), target.float(), reduction="mean"
            )

        return loss

    def setup_pipeline(self):
        scheduler = EulerDiscreteScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            scheduler=scheduler,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            unet=self.accelerator.unwrap_model(self.unet, keep_fp32_wrapper=False),
            vae=self.vae,
            revision=self.config.revision,
            torch_dtype=(
                self.weight_dtype
                if self.accelerator.mixed_precision in ["fp16", "bf16"]
                else torch.float16
            )
        )

        self.pipeline.safety_checker = None
        self.pipeline = self.pipeline.to(self.accelerator.device)
        self.pipeline.set_progress_bar_config(disable=True)

    def setup_model(self):
        # Training SD: freeze vae and text_encoder, train the UNet
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        # Set up LoRA training type (between TLoRA and LoRA) using ClassRegistry()
        attn_processor = processors[self.config.trainer_type]
        linear_layer = linear_layers[self.config.trainer_type]

        self.params_to_optimize = []
        attn_procs = {}
        # Apply LoRA to all attention blocks
        for name in self.unet.attn_processors.keys():
            # get attention dimension
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            # get latent dimension in attention blocks
            if name.startswith("mid_block"): # difference between layers
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            # configurations
            kwargs = {
                "hidden_size": hidden_size,
                "rank":self.config.tensor_train_rank,
                "cross_attention_dim": cross_attention_dim,
            }
            
            # flag original layer to note the processor belongs to
            if isinstance(attn_processor, TensorTrainCrossAttnProcessor):
                # get_layer_by_name: retrieve layer name inside UNet
                kwargs["original_layer"] = get_layer_by_name(self.unet, name.split(".processor")[0])

            # real setup for lora to each attention layer
            attn_procs[name] = attn_processor(**kwargs)

        # Set the attention processors
        self.unet.set_attn_processor(attn_procs) # re-apply newly constructed processors back to attention blocks
        
        # registration with accelerator library
        self.adaptive_layers = AttnProcsLayers(self.unet.attn_processors)
        self.accelerator.register_for_checkpointing(self.adaptive_layers)

        # Set trainable parameters and optimizer
        for name, param in self.adaptive_layers.named_parameters():
            if param.requires_grad == True:
                self.params_to_optimize.append(param) # registration to optimizer
        self.adaptive_layers.train() # set to train mode