import os
import webdataset as wds
import pickle
import struct
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor


import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info, Dataset
from torchvision.transforms import Compose, Resize, Normalize, InterpolationMode


#####################################################
class SquarePad:
    """Transform to pad images to be square."""

    def __call__(self, image):
        _, width, height = image.shape
        max_wh = max(width, height)
        hp = (max_wh - width) // 2
        vp = (max_wh - height) // 2

        # if padding with even number of pixels
        if (max_wh - width) % 2 == 0 and (max_wh - height) % 2 == 0:
            padding = (vp, vp, hp, hp)
        # if vertical padding is needed with odd number of pixels, add one more pixel to the bottom
        elif (max_wh - width) % 2 == 0 and (max_wh - height) % 2 == 1:
            padding = (vp, vp + 1, hp, hp)
        # if horizontal padding is needed with odd number of pixels, add one more pixel to the right
        elif (max_wh - width) % 2 == 1 and (max_wh - height) % 2 == 0:
            padding = (vp, vp, hp, hp + 1)

        return F.pad(image, padding, "constant", 0)


#####################################################
class RGFineTuningWebDataset(IterableDataset):
    def __init__(self, url_list, tokenizer, data_filter_file=None):
        # self.webdataset = wds.WebDataset(url_list).shuffle(1024)
        self.url_list = url_list
        self.webdataset = wds.DataPipeline(
            wds.SimpleShardList(url_list),
            # at this point we have an iterator over all the shards
            wds.shuffle(100),
            wds.tarfile_to_samples(),
            wds.shuffle(1000),
        )

        if data_filter_file is not None:
            self.data_filter = []
            with open(data_filter_file, "r") as file:
                for line in file:
                    self.data_filter.append(line.strip())
            print("Length of data filter:{}".format(len(self.data_filter)))
        else:
            self.data_filter = None
            print("No data filter provided.")

        self.image_transforms = Compose(
            [
                SquarePad(),
                Resize(512, interpolation=InterpolationMode.BILINEAR),
                Normalize([0.5], [0.5]),
            ]
        )

        self.tokenizer = tokenizer

    def __len__(self):
        if self.data_filter is not None:
            return len(self.data_filter)
        else:
            # raise NotImplementedError("Length of dataset is not defined.")
            size_list = [f.split(".tar")[0] + "_size.txt" for f in self.url_list]
            ds_size = 0
            for size_file in size_list:
                with open(size_file, "r") as file:
                    line = file.readline().strip()
                    ds_size += int(line)
            return ds_size

    def wds_item_to_sample(self, item):
        sample = {}

        sample["pixel_values"] = (
            pickle.loads(item["pt_image"]).unsqueeze(0).expand(3, -1, -1)
        )
        sample["pixel_values"] = self.image_transforms(sample["pixel_values"])

        # TODO: change the hard-coded prompt key part to maybe a parameter passed to dataset class
        prompt = item["prompt_metadata"].decode("utf-8")
        prompt_tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        sample["input_ids"] = prompt_tokenized.input_ids.squeeze()
        sample["attention_mask"] = prompt_tokenized.attention_mask.squeeze()
        sample["loss_weights"] = torch.FloatTensor([1.0]).squeeze()

        return sample

    def __iter__(self):
        info = get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        id = info.id if info is not None else 0

        self.source = iter(self.webdataset)
        for i, item in enumerate(self.source):
            if i % num_workers == id:
                # with no data filter, simply yield next item
                if self.data_filter is None:
                    yield self.wds_item_to_sample(item)
                # if data filter is provided, only yield items with dicom_ids in the filter
                elif item["__key__"] in self.data_filter:
                    yield self.wds_item_to_sample(item)


#####################################################
class RGFineTuningImageDirectoryDataset(Dataset):
    """
    A PyTorch Dataset for fine-tuning, loading images from a directory
    and corresponding text prompts from another directory.

    Args:
        image_dir_path (str): Path to the directory containing image files (e.g., .jpg).
        text_dir_path (str): Path to the directory containing text prompt files (e.g., .txt).
        tokenizer (callable): Tokenizer function from Hugging Face transformers.
        data_filter_file (str, optional): Path to a file containing a list of image stems
                                          to include. Each line should be an image stem (e.g., 'image001').
                                          Defaults to None, meaning no filter is applied.
    """
    def __init__(self, image_dir_path, text_dir_path, tokenizer, data_filter_file=None):
        self.image_dir_path = image_dir_path
        self.text_dir_path = text_dir_path
        self.tokenizer = tokenizer

        # Initialize image transformations
        self.image_transforms = Compose(
            [
                ToTensor(), # Converts PIL Image to Tensor and scales to [0, 1]
                SquarePad(), # Pads the image to a square
                Resize(512, interpolation=InterpolationMode.BILINEAR), # Resizes to 512x512
                Normalize([0.5], [0.5]), # Normalizes to [-1, 1]
            ]
        )

        # Collect all image and text file paths
        all_image_files = sorted([f for f in os.listdir(image_dir_path) if f.lower().endswith(('.jpg', '.jpeg'))])
        
        self.samples = [] # List of (image_path, text_path, image_stem) tuples

        for image_filename in all_image_files:
            image_stem = os.path.splitext(image_filename)[0]
            image_path = os.path.join(image_dir_path, image_filename)
            text_path = os.path.join(text_dir_path, image_stem + ".txt")

            if os.path.exists(text_path):
                self.samples.append((image_path, text_path, image_stem))
            else:
                print(f"Warning: No corresponding text file found for image: {image_filename}. Skipping.")

        # Load data filter if provided
        self.data_filter = None
        if data_filter_file is not None:
            self.data_filter = set() # Use a set for efficient lookup
            with open(data_filter_file, "r") as file:
                for line in file:
                    self.data_filter.add(line.strip())
            print(f"Length of data filter: {len(self.data_filter)}")
            
            # Filter samples based on data_filter
            self.samples = [
                (img_p, txt_p, stem) for img_p, txt_p, stem in self.samples
                if stem in self.data_filter
            ]
            print(f"Dataset size after filter: {len(self.samples)}")
        else:
            print("No data filter provided.")
        
        if not self.samples:
            raise ValueError("No valid image-text pairs found after initialization/filtering. Check paths and filter.")


    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing 'pixel_values', 'input_ids',
                  'attention_mask', and 'loss_weights'.
        """
        # Retrieve paths and stem for the current index
        image_path, text_path, image_stem = self.samples[idx]
        
        sample = {}

        # 1. Load and transform image
        image = np.array(Image.open(image_path).convert("RGB"))
        sample["pixel_values"] = self.image_transforms(image)

        # 2. Load and tokenize text prompt
        with open(text_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        prompt_tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        sample["input_ids"] = prompt_tokenized.input_ids.squeeze()
        sample["attention_mask"] = prompt_tokenized.attention_mask.squeeze()

        # 3. Add loss weights (as in the original WebDataset class)
        sample["loss_weights"] = torch.FloatTensor([1.0]).squeeze()

        # Optionally, include the image_stem for debugging or external use
        # sample["image_stem"] = image_stem

        return sample

#####################################################
