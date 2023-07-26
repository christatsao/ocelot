from comet_ml import Experiment

import sys, os

#Our project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.pardir))
sys.path.append(PROJECT_ROOT)

#Local packages loaded from src specifying useful constants, and our custom loader
from util.constants import DATA_PATHS
from util.dataset import OcelotDatasetLoader, OcelotDatasetLoader2, BinaryPixelThreshold
from util.unet import Unet
from util.losses import calc_DiceCELoss
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
from monai.networks.utils import one_hot
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

#First we need to specify some info on our model: we have 3 channels RGB, 1 class: tissue
model = Unet(n_channels=3, n_classes=1)

valtest_transform = A.Compose([ #A.Resize(512,512), #NOTE: BE SURE TO RESIZE ACCORDINGLY + ADD TRANSFORMS BASED ON HOW YOU TRAINED MODEL
                                A.Normalize(mean = 0.0, std=1, always_apply=True),
                                ToTensorV2(),
                                ])

#NOTE: datasetroot and scratchDirData ARE SPECIFIC TO ME ONLY. ADJUST PATHS CORRESPONDINGLY.
datasetroot = "/uufs/chpc.utah.edu/common/home/u6052852/ocelot/data/ocelot2023_v0.1.2"
scratchDirData = '/scratch/general/nfs1/u6052852/REU/Results/RS0/lr0.009/wd0.0001/Data0'

test  = list(pd.read_csv(os.path.join(scratchDirData,'test.csv'),  header=None).loc[:,0])

if model.n_classes == 1:
    multiclass = False
else:
    multiclass =True

testData  = OcelotDatasetLoader2(test,
                                datasetroot,
                                transforms=valtest_transform,
                                multiclass=multiclass) 

#We pass into dataloader provided by torch
test_loader = DataLoader(testData,
                        batch_size=2,
                        num_workers=4)

#NOTE: PLEASE USE YOUR OWN DIRECTORY FOR WHERE YOUR TRAINED MODEL IS SAVED
model.load_state_dict(torch.load('/scratch/general/nfs1/u6052852/REU/Results/RS1/lr0.02/wd0.0001/model.pt'))
model.eval()

idx = 20 #NOTE: YOU CAN ADJUST THIS TO WHATEVER YOU WANT WITIHIN RANGE OF DATA TO SHOW


if not multiclass:
    image = test_loader.dataset[idx][0].unsqueeze(0)
    image = image.float()

    true_mask = test_loader.dataset[idx][1]
    true_mask = true_mask.numpy()

    prediction = torch.sigmoid(model(image))
    threshold = torch.tensor([0.5])
    predicted_mask = (prediction>threshold).float()*1
    predicted_mask = predicted_mask.squeeze(0).squeeze(0).numpy()

    plt.imshow(predicted_mask)
    plt.savefig('test_predicted_mask.png')
    plt.close()

    image = (image.squeeze(0) * 255).to(torch.uint8)
    PILimage = transforms.ToPILImage()(image)
    PILimage.save('test_image.png')

    plt.imshow(true_mask)
    plt.savefig('test_true_mask.png')
    plt.close()

else:
    image = test_loader.dataset[idx][0].unsqueeze(0)
    image = image.float()