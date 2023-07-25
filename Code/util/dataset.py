import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import json
import numpy as np
from torchvision.transforms import ToPILImage
import torch

class OcelotDatasetLoader(Dataset):
    def __init__(self, paths: 'list'=[], dataToLoad = None, image_transforms = None, mask_transforms = None):
        '''
        Args:
            paths: a list of paths to all 5 corresponding data subfolders in order of \
                [cell images, cell annotations, tissue images, tissue masks, metadata]
            dataToReturn: a str corresponding to what kind of data (Cell/cell or Tissue/tissue) \
                is to be returned by the data loader. If none specified, returns everything about data, \
                including x, y coordinates of cell image with respect to tissue image.
        '''
        self.cellIMGPaths = paths[0]
        self.cellIMGFileNames = os.listdir(paths[0])
        self.cellANNPaths = paths[1]
        self.cellANNFileNames = os.listdir(paths[1])

        self.tissIMGPaths = paths[2]
        self.tissIMGFileNames = os.listdir(paths[2])
        self.tissANNPaths = paths[3]
        self.tissANNFileNames = os.listdir(paths[3])

        self.dataToReturn = dataToLoad
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

        self.metadataAbsPath = paths[4]
        with open(self.metadataAbsPath) as jsonFile:
            self.jsonObject = json.load(jsonFile)
            jsonFile.close()

    def __len__(self):
        '''
        Returns:
            Length of dataset based on number of cell images. Ensure ORDERED correspondence of data between subfolders. 
        '''
        return len(os.listdir(self.cellIMGPaths))
    
    def __getitem__(self, idx):
        '''dataset[0]
        Args:
            idx: index of sample of interest. Ensure ORDERED correspondence of data between subfolders. 

        Returns:
            if specified by dataToReturn = Cell/cell or Tissue/tissue, the corresponding image and annotation arrays, else everything.
        '''
        cellImageAbsPath = os.path.join(self.cellIMGPaths, self.cellIMGFileNames[idx])
        tissImageAbsPath = os.path.join(self.tissIMGPaths, self.tissIMGFileNames[idx])
        cellAnnAbsPath = os.path.join(self.cellANNPaths, self.cellANNFileNames[idx])
        tissAnnAbsPath = os.path.join(self.tissANNPaths, self.tissANNFileNames[idx])
        
        cellImage = Image.open(cellImageAbsPath)
        tissImage = Image.open(tissImageAbsPath)
        tissMask = Image.open(tissAnnAbsPath)
        try:
            cellAnn = pd.read_csv(cellAnnAbsPath, delimiter=',').to_numpy()
        except:
            cellAnn = np.empty((0,0,0))

        x_start = self.jsonObject['sample_pairs'][os.path.splitext(self.cellIMGFileNames[idx])[0]]['cell']['x_start']
        x_end = self.jsonObject['sample_pairs'][os.path.splitext(self.cellIMGFileNames[idx])[0]]['cell']['x_end']

        y_start = self.jsonObject['sample_pairs'][os.path.splitext(self.cellIMGFileNames[idx])[0]]['cell']['y_start']
        y_end = self.jsonObject['sample_pairs'][os.path.splitext(self.cellIMGFileNames[idx])[0]]['cell']['y_end']

        x_coord = [x_start, x_end]
        y_coord = [y_start, y_end]

        if self.image_transforms:
            cellImage = self.image_transforms(cellImage)
            tissImage = self.image_transforms(tissImage)
        if self.mask_transforms:
            tissMask = self.mask_transforms(tissMask)

        if self.dataToReturn == 'cell' or self.dataToReturn == 'Cell':
            return cellImage, cellAnn
        
        elif self.dataToReturn == 'tissue' or self.dataToReturn == 'Tissue':
            return tissImage, tissMask

        return cellImage, cellAnn, tissImage, tissMask, x_coord, y_coord
    
class PixelThreshold(object):
  def __init__(self, upper_thresh=None, lower_thresh=None):
    '''Class to apply thresholding to a given tensor'''
    self.lower_thresh = lower_thresh / 255. if lower_thresh else None
    self.upper_thresh = upper_thresh /255. if upper_thresh else None

  def __call__(self, tensor):
    if self.lower_thresh == None and not self.upper_thresh == None:
      return (tensor < self.upper_thresh).to(tensor.dtype)
    
    elif not self.lower_thresh == None and self.upper_thresh == None:
      return (tensor > self.lower_thresh).to(tensor.dtype)

    else:
      return ((tensor > self.lower_thresh) & (tensor < self.upper_thresh)).to(tensor.dtype)


###DEPERECATED______________________________________________________________________________________
def predict_segmentation_img(model, tensor):
    model.eval()
    
    with torch.no_grad():        
        # Perform inference
        predicted_mask = model(tensor.unsqueeze(0))
        predicted_mask = torch.sigmoid(predicted_mask) #TODO: check correctness softmax??
        #predicted_mask = torch.argmax(predicted_mask, dim=1)
        predicted_mask = predicted_mask.squeeze().cpu().numpy() #TODO: SQUEEZE 0 or SQUEEZE?
        print(predicted_mask.shape)
                        
    # Convert the predicted mask to a PIL image
    predicted_mask = Image.fromarray(((predicted_mask > 0.5) * 255).astype(np.uint8)) #TODO: TEST TO SEE IF CORRECT

    return predicted_mask
