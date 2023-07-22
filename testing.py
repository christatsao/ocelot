from comet_ml import Experiment

import sys, os

#Our project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.pardir))
sys.path.append(PROJECT_ROOT)

#Local packages loaded from src specifying useful constants, and our custom loader
from util.constants import DATA_PATHS
from util.dataset import OcelotDatasetLoader, OcelotDatasetLoader2, BinaryPixelThreshold
from util.unet import Unet
from util.evaluate import evaluate
import argparse

#other modules of interest
import torch
from torch.utils.data import DataLoader
import torchmetrics
from PIL import Image
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.losses import DiceCELoss, DiceLoss, MaskedDiceLoss
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from albumentations.augmentations.transforms import Normalize, PixelDropout
from torchvision import transforms


my_device = torch.device(device = 'cuda' if torch.cuda.is_available() else 'cpu')
pin_memory = True if my_device == 'cuda' else False
d_type_f32 = torch.float32
batch_size = 1
learning_rate= 1e-3
weight_decay = 1e-3
nepochs = 10
val_percent=0.1
train_percent = 1 - val_percent

#First we need to specify some info on our model: we have 3 channels RGB, 1 class: tissue
model = Unet(n_channels=3, n_classes=3)

train_transform =   A.Compose([ A.Resize(128,128),
                                A.HorizontalFlip(p=0.5),
                                Normalize(mean=0.0, std=1.0, always_apply=True), #TODO: THIS OR MIN-MAX SCALING?
                                ToTensorV2(),
                                ])

valtest_transform = A.Compose([ A.Resize(128,128),
                                Normalize(mean=0.0, std=1.0,always_apply=True), #TODO: THIS OR MIN-MAX SCALING?
                                ToTensorV2(),
                                ])

datasetroot = "/uufs/chpc.utah.edu/common/home/u6052852/ocelot/data/ocelot2023_v0.1.2"
scratchDirData = '/scratch/general/nfs1/u6052852/REU/Data0'

train = list(pd.read_csv(os.path.join(scratchDirData,'train.csv'), header=None).loc[:,0])
val   = list(pd.read_csv(os.path.join(scratchDirData,'val.csv'),   header=None).loc[:,0])
test  = list(pd.read_csv(os.path.join(scratchDirData,'test.csv'),  header=None).loc[:,0])

train_split = OcelotDatasetLoader2(train,
                                    datasetroot,
                                    transforms=train_transform) 
val_split   = OcelotDatasetLoader2(val,
                                    datasetroot,
                                    transforms=valtest_transform) 
testData  = OcelotDatasetLoader2(test,
                                    datasetroot,
                                    transforms=valtest_transform) 

#We pass into dataloader provided by torch
train_loader = DataLoader(train_split, 
                        batch_size=2, 
                        num_workers=4)
val_loader = DataLoader(val_split, 
                        batch_size=2, 
                        num_workers=4)
test_loader = DataLoader(testData,
                        batch_size=2,
                        num_workers=4)

mask = test_loader.dataset[9][1]
mask = mask.unsqueeze(0)

def one_hot(masks: torch.Tensor, num_classes):
    batch_size, height, width = masks.shape

    one_hot_mask = torch.zeros(batch_size, num_classes, height, width, dtype=masks.dtype, device=masks.device)

    # Fill the one-hot tensor based on the class labels in the mask
    for channel, pixel_value in enumerate([1, 2, 255]):
        one_hot_mask[:, channel, :, :] = (masks == pixel_value)

    return one_hot_mask

print(evaluate(None, model, test_loader, my_device, amp=False))

