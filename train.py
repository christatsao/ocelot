from comet_ml import Experiment

import sys, os

#Our project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.pardir))
sys.path.append(PROJECT_ROOT)

#Local packages loaded from src specifying useful constants, and our custom loader
from util.constants import DATA_PATHS
from util.dataset import OcelotDatasetLoader, PixelThreshold
from util.unet import Unet
from util.evaluate import evaluate
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
import copy
experiment = Experiment(
    api_key="Fl7YrvyVQDhLRYuyUfdHS3oE8",
    project_name="reu-project",
    workspace="joeshmoe03",
)



def tiss_training_loop(args,
        model,
        device,
        experiment,
        #epochs,
        #batch_size:         int = 128,
        #learning_rate:      float = 0.01,
        #val_percent:        float = 0.2,
        #amp:                bool = False,
        #weight_decay:       float = 1e-3,
        #momentum:           float = 0.98,
        #gradient_clipping:  float = 1.0,
        image_transforms = transf.Compose([transf.Resize((128,128)), transf.ToTensor()]),
        mask_transforms = transf.Compose([transf.Resize((128,128)), transf.ToTensor(), PixelThreshold(lower_thresh=1, upper_thresh=255)]),
        filepath = None
    ):
        #Loading our data, performing necessary splits (update with test set in future), and send to loader
        print("Loading Ocelot dataset...")

        training_data = OcelotDatasetLoader(paths=DATA_PATHS,
                                            dataToLoad='Tissue',
                                            image_transforms=image_transforms,
                                            mask_transforms=mask_transforms)
        train_percent = 1 - args.val_percent
        train_N, val_N = [int(train_percent*len(training_data)), 
                        int(args.val_percent*len(training_data))]
        train_split, val_split = torch.utils.data.random_split(training_data, 
                                                            [train_percent, args.val_percent])
        train_loader = DataLoader(train_split, 
                                batch_size=args.batchSize, 
                                num_workers=4)
        val_loader = DataLoader(val_split, 
                                batch_size=args.batchSize, 
                                num_workers=4)
        N_batches_train = train_N/train_loader.batch_size
        N_batches_val = val_N/val_loader.batch_size

        print(f"Found {len(training_data)} data samples.")   
            
        #Initialize optimizer, loss, learning rate, and loss scaling
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learningRate,
                                    momentum=args.momentum,
                                    weight_decay=args.weightDecay,
                                    maximize=False)
        
        criterion = DiceCELoss(sigmoid=True) #TODO: IMPLEMENT BEHAVIOR FOR NON BINARY SEGMENTATION (ensure model.n_classes=1 for now)


        #we use max here as our purpose is to maximize our measured metric (DICE score of 1 is better: more mask similarity)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        
        #Only for AMP. prevents loss of values due to switch between multiple formats
        grad_scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=args.amp)
        model.to(device)
        val_losses = []
        train_losses = []
        best_val = 100000

        #Begin training
        for epoch in range(args.epochs):
            epoch_loss = 0
            model.train()

            with tqdm(total=train_N, desc=f'Epoch {epoch+1}/{args.epochs}') as progress_bar:
                            
                for batch in train_loader:
                    images, true_masks = batch[0], batch[1]
                    assert images.shape[1] == model.n_channels, f"Expected {model.n_channels} channels from image but received {images.shape[1]} channels instead."
                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last if args.amp==True else torch.preserve_format)
                    true_masks = true_masks.to(device=device, dtype=torch.float32)

                    with torch.autocast(device.type if device.type == 'cuda' else 'cpu', enabled=args.amp):
                        infer_masks = model(images)
                        
                        if model.n_classes == 1:
                            loss = criterion(infer_masks, true_masks.float())
                            epoch_loss += loss.detach().cpu().item()
                            #loss += 0.5 * dice_score #TODO: loss += dice score?
                        
                        else:
                            #TODO: EVALUATE CRITERION FOR MULTICLASS SEGMENTATION
                            loss = ...
                            return NotImplementedError

                    optimizer.zero_grad()

                    #Scales w/ AMP enabled from loss and does backprop
                    grad_scaler.scale(loss).backward()

                    #Grad clipping restricts gradient to a range. Research vanishing gradient for more.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
                    
                    #Step the optimizer for new model parameters (keeping grad scaling in mind assuming AMP)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    progress_bar.update(images.shape[0])
                
                #Calculate train loss
                train_loss = epoch_loss/N_batches_train
                train_losses.append(train_loss)

                #Move on to validation loss
                if(epoch%3==0):
                    val_loss = evaluate(args,experiment,model, val_loader, device, epoch,amp=False) #TODO: UPDATE EVALUATION METHOD FOR MULTICLASS

                
                #print(f"Val loss:   {val_loss}")
                #print(f"Train loss: {train_loss}")
                experiment.log_metric('train_loss',train_loss,step=epoch)
                experiment.log_metric('val_loss',val_loss,step=epoch)
                
                scheduler.step()

                val_losses.append(val_loss)

                #TODO: Deepcopy and save the model with WORST/best? val accuracy
                if val_loss < best_val:
                    best_val = val_loss
                    best_trained_model=copy.deepcopy(model.state_dict())
                    torch.save(best_trained_model, os.path.join(filepath,'model.pt'))
        return train_losses, val_losses


def main(args):
    scratchDir ='/scratch/general/nfs1/u6052852/REU_Results' 
    my_device = torch.device(device = 'cuda' if torch.cuda.is_available() else 'cpu')
    pin_memory = True if my_device == 'cuda' else False
    d_type_f32 = torch.float32
    experiment.log_parameters(args)
    #batch_size = 1
    #learning_rate= 1e-3
    #weight_decay = 1e-3
    #nepochs = 10
    #val_percent=0.1
    #train_percent = 1 - val_percent
    scratchDir = os.path.join(scratchDir,'lr'+str(args.learningRate),'wd'+str(args.weightDecay))
    if(os.path.exists(scratchDir)==False):
        os.makedirs(scratchDir)


    image_transforms = transf.Compose([transf.Resize((128,128)), transf.ToTensor()])
    mask_transforms = transf.Compose([transf.Resize((128,128)), transf.ToTensor(), PixelThreshold(lower_thresh=1, upper_thresh=255)])

    #First we need to specify some info on our model: we have 3 channels RGB, 1 class: tissue
    model = Unet(n_channels=args.inputChannel, n_classes=args.outputChannel)

    

    train_score, val_score = tiss_training_loop(args,model,
                            my_device,
                            experiment,
                            mask_transforms=mask_transforms,
                            image_transforms=image_transforms,filepath=scratchDir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Short sample app')
    ## Model Arguments
    parser.add_argument('-ich'              ,type=int  , action="store", dest='inputChannel'     , default=3           )
    parser.add_argument('-och'              ,type=int  , action="store", dest='outputChannel'    , default=1           )
    #parser.add_argument('-filterSize'       ,type=int  , action="store", dest='filterSize'       , default=64          )
    #parser.add_argument('-patchSize'        ,type=int  , action="store", dest='patchSize'        , default=256         )
    #parser.add_argument('-method'           ,type=str  , action="store", dest='method'           , default='baseline'    )
   # parser.add_argument('-norm'             ,type=str  , action="store", dest='norm'             , default='Batch'     )
    #parser.add_argument('-path'             ,type=str  , action="store", dest='path'             , default='./'     )
    #parser.add_argument('-initType'         ,type=int  , action="store", dest='initType'         , default=0           )
    #parser.add_argument('-factorSize'       ,type=int  , action="store", dest='factorSize'       , default=1           )
    #parser.add_argument('-stepEpoch'        ,type=int  , action="store", dest='stepEpoch'        , default=1000        )
    #parser.add_argument('-binaryClass'      ,type=int  , action="store", dest='binaryClass'      , default=0           )
    ## Training Arguments
    parser.add_argument('-g_c'               ,type=float  , action="store", dest='gradient_clipping'         , default=1.0       )
    parser.add_argument('-momen'             ,type=float  , action="store", dest='momentum'       , default=0.98   )
    parser.add_argument('-amp'               ,type=bool  , action="store", dest='amp'          , default=False           )
    parser.add_argument('-lr'               ,type=float, action="store", dest='learningRate'     , default=1e-4        )
    parser.add_argument('-wd'               ,type=float, action="store", dest='weightDecay'      , default=1e-4        )
    parser.add_argument('-nepoch'           ,type=int  , action="store", dest='epochs'            , default=10          )
    parser.add_argument('-batchSize'        ,type=int  , action="store", dest='batchSize'        , default=4           )
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
