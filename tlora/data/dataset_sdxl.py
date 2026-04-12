import os
import json

from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset ####
from torchvision.transforms import (
    Compose, Resize, Normalize, InterpolationMode, ToTensor, RandomCrop, RandomHorizontalFlip, CenterCrop
)

BICUBIC = InterpolationMode.BICUBIC

# tokenizers: list[tokenizers], prompt: str
def tokenize_prompt(tokenizers, prompt):
    text_input_ids_list = []
    for tokenizer in tokenizers:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_list.append(text_inputs.input_ids)
    return text_input_ids_list

# text_encoders: list[module]
# text_input_ids_list: list[tensor]
def encode_tokens(text_encoders, text_input_ids_list):
    prompt_embeds_list = []

    for text_encoder, text_input_ids in zip(text_encoders, text_input_ids_list):
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True, # returns all hidden states (from all transformer layers), not just the last one
            return_dict=False
        )
        # prompt_embeds = (pool_output, last_hidden_state, hidden_states)

        # Note: We are only ALWAYS interested in the pooled output of the final text encoder
        # (batch_size, pooled_dim)
        pooled_prompt_embeds = prompt_embeds[0]
        # (batch_size, seq_len, dim)
        prompt_embeds = prompt_embeds[-1][-2] # the penultimate layer of hidden states, as said in the paper
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1) # faster reshape but continguous, changes nothing if len(shape) == 3
        prompt_embeds_list.append(prompt_embeds)

    # (batch_size, seq_len, dim)
    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1) # concatenate the hidden states from all text encoders at channel-axis
    # (batch_size, pooled_dim)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def compute_time_ids(original_size, crops_coords_top_left, resolution):
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    target_size = torch.tensor([[resolution, resolution]], device=original_size.device)
    target_size = target_size.expand_as(original_size)

    add_time_ids = torch.cat([original_size, crops_coords_top_left, target_size], dim=1)
    return add_time_ids

class ImageDataset(Dataset):
    def __init__(
        self,
        train_data_dir,
        resolution=1024,
        rand=False,
        repeats=100,
        one_image=False,
    ):
        # get file dir
        self.train_data_dir = train_data_dir

        # get file names
        self.data_fnames = [
            os.path.join(r, f) for r, d, fs in os.walk(self.train_data_dir) #os.walk() recursive for inner folders by default, no need to alter
            for f in fs
        ]
        self.data_fnames = sorted(self.data_fnames)
        if one_image: # only select one image
            self.data_fnames = [a for a in self.data_fnames if a.endswith(one_image)]

        # filter out non-image files
        self.data_fnames = [f for f in self.data_fnames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        self.num_images = len(self.data_fnames)
        self._length = self.num_images * repeats

        self.resolution = resolution
        self.rand = rand

    def process_img(self, img): # for SDXL
        #RGB
        image = Image.open(img).convert('RGB')
        w, h = image.size
        crop = min(w, h)
        # data augmentation: crop from rectangle image to square
        if self.rand:
            # scale the image, that the shorter side is to 560 px (Note: a bit hard-coding there)
            image = Resize(560, interpolation=InterpolationMode.BILINEAR, antialias=True)(image)
            # random crop with given resolution
            image = RandomCrop(self.resolution)(image)
            # random horizontal flip
            image = RandomHorizontalFlip()(image)
        else:
            # central cropping
            image = image.crop(((w - crop) // 2, (h - crop) // 2, (w + crop) // 2, (h + crop) // 2))

        # resize to main resolution using LANCZOS resampling filter
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)

        # convert to torch tensor
        input_img = torch.cat([ToTensor()(image)])

        # why two outputs? see __getitem__()
        return input_img, torch.tensor([crop, crop])

    def __len__(self):
        return self._length

    def __getitem__(self, index): #similar to collate_fn
        example = {}
        image_file = self.data_fnames[index % self.num_images]
        input_img, example["original_sizes"] = self.process_img(image_file)
        # assert example["original_sizes"][0] == example["original_sizes"][1], \
        #     'SDXL has a complicated procedure to handle rectangle images. We do not implement it'
        example["crop_top_lefts"] = torch.tensor([0, 0])

        example['image_path'] = image_file
        example['image'] = input_img

        # return a dictionary containing image and its related info.
        return example


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizers,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=1024,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizers = tokenizers

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)

            # to save during runtime as well
            # parents=True: create parent directories if they don't exist
            # exist_ok=True: do not raise an error if the directory already exists
            self.class_data_root.mkdir(parents=True, exist_ok=True)

            # get list of class image paths
            self.class_images_path = list(self.class_data_root.iterdir())

            # set max number of class images to use, if class_num is specified
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)

            # set dataset length to the max of instance and class images, so that we can loop through both in __getitem__
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        # image preprocessing pipeline setup
        self.image_transforms = Compose(
            [
                Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=True),
                CenterCrop(size) if center_crop else RandomCrop(size),
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        # get instance image
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        example["original_size"] = torch.tensor(instance_image.size)
        assert instance_image.size[0] == instance_image.size[1], \
            'SDXL has a complicated procedure to handle rectangle images. We do not implement it'
        example["crop_top_left"] = torch.tensor([0, 0])

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"], example["instance_prompt_ids_2"] = tokenize_prompt(self.tokenizers, self.instance_prompt)

        # get class image
        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"], example["class_prompt_ids_2"] = tokenize_prompt(self.tokenizers, self.class_prompt)

        return example


# custom collate for DreamBooth
def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    input_ids_2 = [example["instance_prompt_ids_2"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    original_sizes = [example["original_size"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        input_ids_2 += [example["class_prompt_ids_2"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        original_sizes += [example["original_size"] for example in examples]
        crop_top_lefts += [example["crop_top_left"] for example in examples]

    # this is just to resize
    pixel_values = torch.stack(pixel_values)
    original_sizes = torch.stack(original_sizes)
    crop_top_lefts = torch.stack(crop_top_lefts)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    input_ids_2 = torch.cat(input_ids_2, dim=0)

    batch = {
        "input_ids": input_ids,
        "input_ids_2": input_ids_2,
        "pixel_values": pixel_values,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts
    }
    return batch

class CustomDataset(Dataset):
    def __init__(self,
        data_root,
        data_prompt_path,
        prompt_and_class_path="prompts",
        tokenizers,
        resolution=1024,
        center_crop=False
    ):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError("Data root doesn't exists.")
        
        # get list of prompts and class names from "prompts_and_classes.txt"
        self.data_fnames = 


        self.num_images = len(self.data_fnames)
        self._length = self.num_images
        with open(data_prompt_path, 'r') as f:
            data_prompts = json.load(f)
        self.prompts = [data_prompts[img.name] for img in self.data_fnames]
        # image preprocessing pipeline setup
        self.image_transforms = Compose(
            [
                Resize(resolution, interpolation=InterpolationMode.BILINEAR, antialias=True),
                CenterCrop(resolution) if center_crop else RandomCrop(resolution),
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )
