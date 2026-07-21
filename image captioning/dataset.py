"""
dataset.py
----------
PyTorch Dataset for loading (image, caption) pairs from a Flickr8k/Flickr30k
style captions file (CSV with columns: image,caption). Also provides the
image transforms and a collate function that pads captions within a batch.
"""

import os
import pandas as pd
from PIL import Image

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

import config
from vocabulary import Vocabulary


# ImageNet normalization stats -> required because the encoder is pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class CaptionDataset(Dataset):
    def __init__(self, image_dir, captions_file, vocab: Vocabulary, transform=None):
        self.image_dir = image_dir
        self.df = pd.read_csv(captions_file)
        assert {"image", "caption"}.issubset(self.df.columns), (
            "captions.txt must have columns named 'image' and 'caption'"
        )
        self.images = self.df["image"].tolist()
        self.captions = self.df["caption"].tolist()
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        numericalized = [self.vocab.stoi["<SOS>"]]
        numericalized += self.vocab.numericalize(caption)
        numericalized.append(self.vocab.stoi["<EOS>"])
        numericalized = numericalized[: config.MAX_CAPTION_LEN]

        return image, torch.tensor(numericalized, dtype=torch.long)


class CapCollate:
    """Pads all captions in a batch to the length of the longest one."""

    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)

        captions = [item[1] for item in batch]
        lengths = torch.tensor([len(c) for c in captions], dtype=torch.long)
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)

        return images, captions, lengths


def build_vocab_from_captions(captions_file: str) -> Vocabulary:
    df = pd.read_csv(captions_file)
    vocab = Vocabulary(freq_threshold=config.FREQ_THRESHOLD)
    vocab.build_vocabulary(df["caption"].tolist())
    return vocab
