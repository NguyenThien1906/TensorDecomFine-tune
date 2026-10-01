import argparse
import sys
from argparse import Namespace

from tlora.trainer_sdxl import trainers

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a inference script."
    )
    parser.add_argument("--trainer_type", type=str, required=True)
    parser.add_argument("--trainer_class", type=str, required=True)
    parser.add_argument("--project_name", type=str, default="tlora")
    parser.add_argument("--wandb_api_key", type=str, required=False, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--revision", type=str, default=None)
    
    parser.add_argument("--num_train_epochs", type=int, default=1000, required=True)
    parser.add_argument("--checkpointing_steps", type=int, default=200, required=True)
    parser.add_argument("--minimum_checkpointing_steps", type=int, default=600, required=False)
    parser.add_argument("--start_checkpoint", type=str, default=None, required=False)

    #parser.add_argument("--train_data_dir", type=str, default=None, required=False)
    parser.add_argument("--train_data_csv", type=str, default=None, required=False, help="Path to CSV file containing training data. Required if not using --train_data_dir.")
    parser.add_argument("--test_data_csv", type=str, default=None, required=False, help="Path to CSV file containing test data. Required if not using --train_data_dir.")
    parser.add_argument("--image_dir", type=str, default=None, required=False, help="Path to the directory containing the images.")
    parser.add_argument("--batch_size", type=int, default=1) #link this to trainer_sdxl.py::LoRATrainerSDXL, 1 is single image
    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, required=False)

    parser.add_argument("--class_data_dir", type=str, default=None, required=False, help="A folder containing the training data of class images.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument("--with_prior_preservation", default=False, action="store_true", help="Flag to add prior preservation loss.")

    parser.add_argument("--class_name", type=str, required=False)
    parser.add_argument("--placeholder_token", type=str, required=False)
    parser.add_argument("--validation_prompts", type=str, default=None)
    parser.add_argument("--num_val_imgs_per_prompt", type=int, default=5)

    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--min_rank", type=int, default=1)
    parser.add_argument("--sig_type", type=str, required=False, default="last", choices=["principal", "last", "middle"])
    parser.add_argument("--alpha_rank_scale", type=float, default=1.0)

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")

    parser.add_argument("--one_image", type=str, default=None)

    parser.add_argument("--early_stopping_patience", type=int, default=10000, help="Number of epochs with no improvement after which training will be stopped.")

    parser.add_argument("--tucker_config_json", type=str, default=None, help="Path to JSON config file for Tucker decomposition ranks. Required if using TuckerLoRALinearLayer.")
    parser.add_argument("--tucker_mode", type=int, default=4, help="The mode of the Tucker decomposition. Only used if using TuckerLoRALinearLayer.")
    parser.add_argument("--tensor_train_rank", type=int, default=4, help="The rank for Tensor Train decomposition. Only used if using TensorTrainLinearLayer.")

    parser.add_argument("--resume_training", default=False, action="store_true", help="Whether training should be resumed from the latest checkpoint in output_dir.")
    parser.add_argument("--resume_config_path", type=str, default=None, help="Path to hparams.yml for resuming training. Required if --resume_training is set.")

    args = parser.parse_args()
    args.argv = [sys.executable] + sys.argv

    return args


def main(args):
    if args.resume_training:
        if args.resume_config_path is None:
            raise ValueError("resume_config_path must be provided when resume_training is set.")
        print(f"Resuming training from config: {args.resume_config_path}")

        with open(args.resume_config_path, "r", encoding="utf-8") as config_file:
            import yaml
            config = yaml.safe_load(config_file)

        config["resume_training"] = True
        config["resume_config_path"] = args.resume_config_path
        config["num_train_epochs"] = int(args.num_train_epochs)
        config["checkpointing_steps"] = int(args.checkpointing_steps)
        config["minimum_checkpointing_steps"] = int(args.minimum_checkpointing_steps)
        config["start_checkpoint"] = int(args.start_checkpoint)

        config_args = Namespace(**config)

        trainer = trainers[config["trainer_class"]](config_args)
    else:
        trainer = trainers[args.trainer_class](args)
    trainer.setup()
    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    main(args)
