from util.evaluate import evaluate
from util.dataset import OcelotDatasetLoader2
from util.unet import Unet
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import os
import pandas as pd

my_device = torch.device(device = 'cuda' if torch.cuda.is_available() else 'cpu')
pin_memory = True if my_device == 'cuda' else False
d_type_f32 = torch.float32

datasetroot = "/uufs/chpc.utah.edu/common/home/u6052852/ocelot/data/ocelot2023_v0.1.2"
scratchDirData = '/scratch/general/nfs1/u6052852/REU/Results/RS1/lr0.005/wd0.001/Data1'

test  = list(pd.read_csv(os.path.join(scratchDirData,'test.csv'),  header=None).loc[:,0])

#First we need to specify some info on our model: we have 3 channels RGB, 1 class: tissue
model = Unet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load('/scratch/general/nfs1/u6052852/REU/Results/RS1_Checkpoint/lr0.005/wd0.001/model.pt'))


if model.n_classes == 1:
    multiclass = False
else:
    multiclass =True

valtest_transform = A.Compose([ A.Normalize(mean = 0.0, std=1, always_apply=True),
                                ToTensorV2(),
                                ])

testData  = OcelotDatasetLoader2(test,
                                    datasetroot,
                                    transforms=valtest_transform,
                                    multiclass=multiclass) 

#We pass into dataloader provided by torch
test_loader = DataLoader(testData,
                        batch_size=2,
                        num_workers=4)

print(evaluate(None, model, test_loader, my_device))