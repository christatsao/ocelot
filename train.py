from comet_ml import Experiment

import sys, os

#Our project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.pardir))
sys.path.append(PROJECT_ROOT)

#Local packages loaded from src specifying useful constants, and our custom loader
from util.constants import DATA_PATHS
from util.dataset import OcelotDatasetLoader, OcelotDatasetLoader2
from util.unet import Unet
from util.calc_loss import calc_DiceCEloss
from util.test import test

import argparse

#other modules of interest
import torch
from torch.utils.data import DataLoader
import torchmetrics
from torchvision import transforms as transf
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

experiment = Experiment(
    api_key="Fl7YrvyVQDhLRYuyUfdHS3oE8",
    project_name="reu-project",
    workspace="joeshmoe03",
)

def tiss_training_loop(args,
        train_loader,
        val_loader,
        model,
        device,
        experiment,
        train,
        filepath = None
    ):                
        #Initialize optimizer, loss, learning rate, and loss scaling
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learningRate,
                                    momentum=args.momentum,
                                    weight_decay=args.weightDecay)
        
        #criterion = DiceCELoss(sigmoid=True)
        if model.n_classes == 1:
            criterion = DiceCELoss(sigmoid=True)
        else:
            criterion =  DiceCELoss(softmax=True, to_onehot_y=True)

        #we use max here as our purpose is to maximize our measured metric (DICE score of 1 is better: more mask similarity)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        #Only for AMP. prevents loss of values due to switch between multiple formats
        model.to(device)
        val_losses = []
        train_losses = []
        best_val = 2.00
        N_batches_train = len(train)/args.batchSize

        #Begin training
        for epoch in range(args.epochs):
            epoch_loss = 0
            model.train()

            with tqdm(total=len(train), desc=f'Epoch {epoch+1}/{args.epochs}') as progress_bar:
                            
                for batch in train_loader:
                    images, true_masks = batch[0], batch[1]
                    assert images.shape[1] == model.n_channels, f"Expected {model.n_channels} channels from image but received {images.shape[1]} channels instead."
                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last if args.amp==True else torch.preserve_format)
                    true_masks = true_masks.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.unsqueeze(1)

                    with torch.autocast(device.type if device.type == 'cuda' else 'cpu', enabled=args.amp):
                        
                        infer_masks = model(images) 
                        loss = criterion(infer_masks, true_masks.float())
                        epoch_loss += torch.sum(loss).detach().cpu().item()

                    optimizer.zero_grad()
                    loss.backward()
                    
                    #Step the optimizer for new model parameters (keeping grad scaling in mind assuming AMP)
                    optimizer.step()
                    progress_bar.update(images.shape[0])
                
                #Calculate train loss
                train_loss = (epoch_loss/N_batches_train)/model.n_classes
                train_losses.append(train_loss)

                #Move on to validation loss
                val_loss = calc_DiceCEloss(args, 
                                    model, 
                                    val_loader, 
                                    device, 
                                    amp=args.amp) #TODO: UPDATE EVALUATION METHOD FOR MULTICLASS

                #Log metrics on Comet ML
                experiment.log_metric('train_loss', train_loss, step=epoch)
                experiment.log_metric('val_loss', val_loss, step=epoch)
                
                scheduler.step()
                val_losses.append(val_loss)

                #Save the best performing model
                if val_loss < best_val:
                    best_val = val_loss
                    best_trained_model=copy.deepcopy(model.state_dict())
                    torch.save(best_trained_model, os.path.join(filepath, 'model.pt'))

        return train_losses, val_losses

def main(args):
    experiment.log_parameters(args)

    scratchDir ='/scratch/general/nfs1/u6052852/REU' #NOTE: PASTE IN YOUR OWN scratchDir
    scratchDir = os.path.join(scratchDir,'Results','RS'+str(args.resample),'lr'+str(args.learningRate),'wd'+str(args.weightDecay))
    if(os.path.exists(scratchDir)==False):
        os.makedirs(scratchDir)

    datasetroot = os.path.join(PROJECT_ROOT, "/ocelot/data/ocelot2023_v0.1.2")

    scratchDirData = os.path.join(scratchDir,'Data'+str(args.resample))
    if(os.path.exists(scratchDirData)==False):
        os.makedirs(scratchDirData)
        table = pd.read_csv(os.path.join(datasetroot,'data.csv'),header=None)
        trainData, testData = train_test_split(table,test_size=0.2)
        train, val = train_test_split(trainData,test_size=0.2)
        testData.to_csv(os.path.join(scratchDirData,'test.csv'),  header=None, index=False)
        train.to_csv   (os.path.join(scratchDirData,'train.csv'), header=None, index=False)
        val.to_csv     (os.path.join(scratchDirData,'val.csv'),   header=None, index=False)

    #Create a list of samples to train, validate, and test on. Resampling can generate a new combination of data
    train = list(pd.read_csv(os.path.join(scratchDirData,'train.csv'), header=None).loc[:,0])
    val   = list(pd.read_csv(os.path.join(scratchDirData,'val.csv'),   header=None).loc[:,0])
    test  = list(pd.read_csv(os.path.join(scratchDirData,'test.csv'),  header=None).loc[:,0])

    #Some device specifications
    my_device = torch.device(device = 'cuda' if torch.cuda.is_available() else 'cpu')
    pin_memory = True if my_device == 'cuda' else False

    #specify datatype to work with
    d_type_f32 = torch.float32

    #First we need to specify some info on our model: we have 3 channels RGB, 1 class: tissue
    model = Unet(n_channels=args.inputChannel, n_classes=args.outputChannel)
    #model.load_state_dict(torch.load('/scratch/general/nfs1/u6052852/REU/Results/RS0/TrainingCheckpoints/lr0.009/wd0.0001/model.pt'))

    #The transformations we are applying to the data that we are training or validating/testing on. 
    #Training data undergoes data augmentation for model performance improvements with such limited data.
    train_transform = A.Compose([#A.Resize(128,128),
                                A.ColorJitter(p=0.4),
                                #A.Affine(keep_ratio=True, p=0.1),   #KEEP?
                                A.Flip(p=0.5),
                                A.ToGray(p=0.5),          
                                #A.Equalize(p=0.2),                  #KEEP?
                                A.Blur(blur_limit=2, p=0.2),
                                A.ElasticTransform(p=0.2),
                                A.GaussNoise(p=0.2),
                                A.HorizontalFlip(p=0.5),
                                A.RandomRotate90(p=0.5), #TODO: MIN-MAX INSTEAD OF NORMALIZATION? REMOVE RESIZING WHEN DONE. AVOID RESIZE.
                                A.Normalize(mean = 0.0, std=1, always_apply=True),
                                ToTensorV2()])
    valtest_transform = A.Compose([#A.Resize(128,128),
                                A.Normalize(mean = 0.0, std=1, always_apply=True),
                                ToTensorV2()])           #TODO: MIN-MAX INSTEAD OF NORMALIZATION? REMOVE RESIZING WHEN DONE.

    if model.n_classes > 1:
        multiclass = True
    else:
        multiclass = False

    #We perform the necessary train/val/test loading based on our resampling
    train_split = OcelotDatasetLoader2(train,
                                       datasetroot,
                                       transforms=train_transform,
                                       multiclass=multiclass) 
    val_split   = OcelotDatasetLoader2(val,
                                       datasetroot,
                                       transforms=valtest_transform,
                                       multiclass=multiclass) 
    testData  = OcelotDatasetLoader2(test,
                                     datasetroot,
                                     transforms=valtest_transform,
                                     multiclass=multiclass) 
    
    #We pass into dataloader provided by torch
    train_loader = DataLoader(train_split, 
                             batch_size=args.batchSize, 
                            num_workers=4)
    val_loader = DataLoader(val_split, 
                            batch_size=args.batchSize, 
                            num_workers=4)
    test_loader = DataLoader(testData,
                             batch_size=args.batchSize,
                             pin_memory=4)

    #Start the actual training loop
    train_score, val_score = tiss_training_loop(args,
                                                train_loader,
                                                val_loader,
                                                model,
                                                my_device,
                                                experiment,
                                                train,
                                                filepath=scratchDir)
    
    test_score = calc_DiceCEloss(args, 
                                model, 
                                test_loader, 
                                device=my_device, 
                                amp=args.amp)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Short sample app')
    ## Model Arguments
    parser.add_argument('-ich'              ,type=int  , action="store", dest='inputChannel'     , default=3           )
    parser.add_argument('-och'              ,type=int  , action="store", dest='outputChannel'    , default=1           )
    parser.add_argument('-resample'         ,type=int  , action="store", dest='resample'         , default=0           )
    #parser.add_argument('-patchSize'        ,type=int  , action="store", dest='patchSize'        , default=256         )
    #parser.add_argument('-method'           ,type=str  , action="store", dest='method'           , default='baseline'    )
    #parser.add_argument('-norm'             ,type=str  , action="store", dest='norm'             , default='Batch'     )
    #parser.add_argument('-path'             ,type=str  , action="store", dest='path'             , default='./'     )
    #parser.add_argument('-initType'         ,type=int  , action="store", dest='initType'         , default=0           )
    #parser.add_argument('-factorSize'       ,type=int  , action="store", dest='factorSize'       , default=1           )
    #parser.add_argument('-stepEpoch'        ,type=int  , action="store", dest='stepEpoch'        , default=1000        )
    #parser.add_argument('-binaryClass'      ,type=int  , action="store", dest='binaryClass'      , default=0           )
    ## Training Arguments
    #parser.add_argument('-g_c'               ,type=float  , action="store", dest='gradient_clipping'         , default=1.0       )
    parser.add_argument('-momen'             ,type=float  , action="store", dest='momentum'       , default=0.98   )
    parser.add_argument('-amp'               ,type=bool  , action="store", dest='amp'          , default=False           )
    parser.add_argument('-lr'               ,type=float, action="store", dest='learningRate'     , default=1e-4        )
    parser.add_argument('-wd'               ,type=float, action="store", dest='weightDecay'      , default=1e-4        ) #NOTE: 1e-4 default
    parser.add_argument('-nepoch'           ,type=int  , action="store", dest='epochs'            , default=100          )
    parser.add_argument('-batchSize'        ,type=int  , action="store", dest='batchSize'        , default=2           )
    #parser.add_argument('-sourcedataset'    ,type=str  , action="store", dest='sourcedataset'          , default='crag'      )
    #parser.add_argument('-targetdataset'    ,type=str  , action="store", dest='targetdataset'          , default='glas'      )
    #parser.add_argument('-modelType'        ,type=str  , action="store", dest='modelType'        , default='unet'      )
    ## For running Inference Only
    parser.add_argument('-val_percent'   ,type=float, action="store", dest='val_percent'    , default=0.2  )
    #parser.add_argument('-targetdataPercentage'   ,type=float, action="store", dest='targetdataPercentage'    , default=1.0  )
    #parser.add_argument('-resnetModel'      ,type=int  , action="store", dest='ResNetModel'        , default=50     )
    #parser.add_argument('-resentInit'       ,type=str  , action="store", dest='ResNetInit'        , default='Random'      )
    #parser.add_argument('-sslType'          ,type=str  , action="store", dest='sslType'        , default='BT'      )
    #parser.add_argument('-decoderFineTuningOnly'        ,type=int  , action="store", dest='decoderFineTuningOnly'        , default=0     )
    #parser.add_argument('-opt'              ,type=int  , action="store", dest='opt'             , default=0  )
    #parser.add_argument('-noP'              ,type=int  , action="store", dest='number_of_patches' , default=2   )
    #parser.add_argument('-resampleData'     ,type=int  , action="store", dest='resample'          , default=0)
    #parser.add_argument('-debug' ,type=int  , action="store", dest='debug'  , default=0)
    args = parser.parse_args()
    main(args)
