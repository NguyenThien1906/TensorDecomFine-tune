import os
import json
import random

from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset ####
from torchvision.transforms import (
    Compose, Resize, Normalize, InterpolationMode, ToTensor, RandomCrop, RandomHorizontalFlip, CenterCrop
)

BASE_PROMPT = "a photo of {0} {1}"
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

        # convert to torch tensor and concatenate between channels
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
        config,
        image_list,
        tokenizers,
        prompt_and_class_path="prompts_and_classes.json",
        resolution=1024,
        rand=False
    ):
        self.resolution=resolution
        self.config = config
        self.image_list = image_list # list of images in this dataset
        self.tokenizers = tokenizers
        self.prompt_and_class_path = Path(prompt_and_class_path)
        if not self.image_list:
            raise ValueError("Image list is empty.")
        if not self.prompt_and_class_path.exists():
            raise ValueError("Prompt and class file doesn't exists.")
        self.rand = rand

        # get list of prompts and class names from "prompts_and_classes.json"
        self.data = None
        with open(self.prompt_and_class_path, 'r') as f:
            self.data = json.load(f)

        if self.data is None:
            raise ValueError("Failed to load prompts and classes from the file.")

        # calculate num of images
        self.num_images = 0
        for subject_name in self.image_list.keys():
            self.num_images += len(self.image_list[subject_name])
        
        # The main intention of this dataset is to have a random prompt assigned for each image per pass
        self._length = self.num_images

    def process_img(self, image):
        #RGB
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

    def getimageidx(self, idx):
        if idx >= self.num_images or idx < 0:
            raise IndexError(f"Index out of range: {idx}")
        if self.data is None:
            raise ValueError("Data is not loaded.")
        for subject_name in self.data["classes"].keys():
            num_subject_images = len(self.image_list[subject_name])
            if idx < num_subject_images:
                return subject_name, self.image_list[subject_name][idx]
            else:
                idx -= num_subject_images
        raise IndexError(f"Index out of range: {idx}")

    def __getitem__(self, index):
        example = {}

        # retrieval
        subject_name, image_path = self.getimageidx(index)
        image = Image.open(image_path).convert('RGB')

        # # get prompts from class name and sample one randomly
        class_name = self.data["classes"][subject_name]
        # class_type = self.data["class_types"][class_name]
        # prompt_type = self.data["prompt_types"][class_type]
        # prompt_list = self.data["prompts"][prompt_type]
        # prompt_list = prompt_list + [BASE_PROMPT]
        # prompt = random.choice(prompt_list)

        # by DreamBooth paper, prompt used is only the base prompt.
        prompt = BASE_PROMPT.format("", class_name).replace("  ", " ")

        # image
        example["image"], example["original_sizes"] = self.process_img(image)
        example["crop_top_left"] = torch.tensor([0, 0])

        # prompt tokenization
        example["prompt_ids"], example["prompt_ids_2"] = tokenize_prompt(self.tokenizers, prompt)

        return example

    def __len__(self):
        return self._length
    
    def number_images(self):
        return self.num_images

def collate_custom(examples):
    input_ids = [example["prompt_ids"] for example in examples]
    input_ids_2 = [example["prompt_ids_2"] for example in examples]
    image = [example["image"] for example in examples]
    original_sizes = [example["original_sizes"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]

    # this is just to resize
    image = torch.stack(image)
    original_sizes = torch.stack(original_sizes)
    crop_top_lefts = torch.stack(crop_top_lefts)
    image = image.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    input_ids_2 = torch.cat(input_ids_2, dim=0)

    batch = {
        "input_ids": input_ids,
        "input_ids_2": input_ids_2,
        "image": image,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts
    }
    return batch

import io
import pandas as pd
import ast
class CsvDataset:
    def __init__(self, csv_file, image_dir, tokenizers, resolution=512, augment_acc=(10, 10)):
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.tokenizers = tokenizers
        self.resolution = resolution
        self.augment_acc = augment_acc
        self.load_csv()

    def load_csv(self):
        data = pd.read_csv(self.csv_file)
        self.image_paths = data["image"].tolist()
        self.image_paths = [os.path.join(self.image_dir, i) for i in self.image_paths]
        self.prompts = data["prompt"].tolist()
        self.data_len = len(self.image_paths)

    def process_img(self, image):
        # the main purpose is to crop down to the correct resolution
        
        # original sizes
        og_w, og_h = image.size

        # resize for augmentation
        aug_w = int(og_w * (1 + self.augment_acc[0] / 100))
        aug_h = int(og_h * (1 + self.augment_acc[1] / 100))
        image = Resize(min(aug_w, aug_h),
            interpolation=InterpolationMode.BILINEAR, antialias=True)(image)

        # crop
        x, y, h, w = RandomCrop.get_params(image, output_size=(self.resolution, self.resolution))
        image = image.crop((x, y, x + w, y + h))

        # final resize
        image = image.resize((self.resolution, self.resolution),
            Image.Resampling.LANCZOS)

        # crop info
        crop_top_left = torch.tensor([x, y])

        return torch.cat([ToTensor()(image)]), torch.tensor([og_w, og_h]), crop_top_left

    def __len__(self):
        return self.data_len


    def __getitem__(self, index):
        example = {}

        # retrieval
        if index >= self.data_len or index < 0:
            raise IndexError(f"Index out of range: {index}")
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        prompt = self.prompts[index]

        # image
        example["image"], example["original_sizes"], example["crop_top_left"] = self.process_img(image)
        #example["image"] = example["image"].to(self.weight_dtype)

        # prompt tokenization
        example["prompt_ids"], example["prompt_ids_2"] = tokenize_prompt(self.tokenizers, prompt)

        return example