import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as T
from torchvision import models
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser

"""
paper for sex classifier: https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(23)00032-4/fulltext
code: https://github.com/biomedia-mira/chexploration/blob/main/prediction/chexpert.sex.py
model weights: https://imperialcollegelondon.app.box.com/s/bq87wkuzy14ctsyf8w3hcikwzu8386jj/folder/149075016249
"""


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


class SexModelDenseNet(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = models.densenet121(pretrained=True)
        freeze_model(self.model)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer

    def unpack_batch(self, batch):
        return batch["image"], batch["label"]

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        loss = F.cross_entropy(out, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log("train_loss", loss)
        grid = torchvision.utils.make_grid(
            batch["image"][0:4, ...], nrow=2, normalize=True
        )
        self.logger.experiment.add_image("images", grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log("test_loss", loss)


class SexModelResNet(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = models.resnet34(pretrained=True)
        # freeze_model(self.model)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer

    def unpack_batch(self, batch):
        return batch["image"], batch["label"]

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        loss = F.cross_entropy(out, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log("train_loss", loss)
        grid = torchvision.utils.make_grid(
            batch["image"][0:4, ...], nrow=2, normalize=True
        )
        self.logger.experiment.add_image("images", grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log("test_loss", loss)


def get_sex_model_imperial(checkpoint_path):
    model = SexModelResNet.load_from_checkpoint(checkpoint_path, num_classes=2)

    model.eval()
    return model
