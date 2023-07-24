import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import json
import numpy as np
from torchvision.transforms import ToPILImage
import torch
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.transforms import Normalize, PixelDropout
import cv2

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

class OcelotDatasetLoader2(Dataset):
    def __init__(self, fileList, datasetroot, transforms=None, multiclass=False):
        ''''''
        self.fileList = fileList
        self.transforms = transforms
        self.root = datasetroot
        self.multiclass = multiclass

    def __len__(self):
        '''
        Returns:
            Length of dataset based on number of cell images. Ensure ORDERED correspondence of data between subfolders. 
        '''
        return len(self.fileList)
    
    def __getitem__(self, idx):
        '''
        Args:
            idx: index of sample of interest. Ensure ORDERED correspondence of data between subfolders. 
        Returns:
        '''
        #cellImageAbsPath = os.path.join(self.cellIMGPaths, self.cellIMGFileNames[idx])
        tissImageAbsPath = os.path.join(self.root,'images','train','tissue')
        tissAnnAbsPath = tissImageAbsPath.replace('images','annotations')
        #cellImageAbsPath = tissImageAbsPath.replace('tissue','cell')

        image_number = self.fileList[idx]
        #cellImage = Image.open(cellImageAbsPath)
        tissImage = cv2.imread(os.path.join(tissImageAbsPath,image_number))
        tissMask = cv2.imread(os.path.join(tissAnnAbsPath,image_number.replace('.jpg','.png')), 0)
        
        #try:
        #    cellAnn = pd.read_csv(cellAnnAbsPath, delimiter=',').to_numpy()
        #except:
        #    cellAnn = np.empty((0,0,0))

        #x_start = self.jsonObject['sample_pairs'][os.path.splitext(self.cellIMGFileNames[idx])[0]]['cell']['x_start']
        #x_end = self.jsonObject['sample_pairs'][os.path.splitext(self.cellIMGFileNames[idx])[0]]['cell']['x_end']

        #y_start = self.jsonObject['sample_pairs'][os.path.splitext(self.cellIMGFileNames[idx])[0]]['cell']['y_start']
        #y_end = self.jsonObject['sample_pairs'][os.path.splitext(self.cellIMGFileNames[idx])[0]]['cell']['y_end']

        #x_coord = [x_start, x_end]
        #y_coord = [y_start, y_end]

        if self.transforms is not None:
            #cellImage = self.image_transforms(cellImage)
            tsample = self.transforms(image=tissImage, mask=tissMask)
            tissImage = tsample['image']
            tissMask = tsample['mask']

        #If any of the transforms seeks to convert the data to a tensor, our given mask should be thresholded such that we only have integers [0,1]
        #This is necessary for dice score calculations
        if any(transform.__class__ == ToTensorV2 for transform in self.transforms) and self.multiclass == False:
            threshold = BinaryPixelThreshold(lower_thresh=1, upper_thresh=255) #TODO: NOT IDEAL BUT WORKS FOR NOW
            tissMask = threshold(tissMask)
        elif any(transform.__class__ == ToTensorV2 for transform in self.transforms) and self.multiclass == True:
            tissMask = torch.where(tissMask == 1, 0, tissMask)
            tissMask = torch.where(tissMask == 2, 1, tissMask) 
            tissMask = torch.where(tissMask == 255, 2, tissMask)

        #if self.dataToReturn == 'cell' or self.dataToReturn == 'Cell':
        #    return cellImage, cellAnn
        #
        #elif self.dataToReturn == 'tissue' or self.dataToReturn == 'Tissue':
        #    return tissImage, tissMask

        #return cellImage, cellAnn, tissImage, tissMask, x_coord, y_coord
        return tissImage, tissMask

class BinaryPixelThreshold(object):
  def __init__(self, upper_thresh=None, lower_thresh=None):
    '''Class to apply thresholding to a given tensor'''
    self.lower_thresh = lower_thresh if lower_thresh else None
    self.upper_thresh = upper_thresh if upper_thresh else None

  def __call__(self, tensor):
    if self.lower_thresh == None and not self.upper_thresh == None:
      return (tensor < self.upper_thresh).to(tensor.dtype)
    
    elif not self.lower_thresh == None and self.upper_thresh == None:
      return (tensor > self.lower_thresh).to(tensor.dtype)

    else:
      return ((tensor > self.lower_thresh) & (tensor < self.upper_thresh)).to(tensor.dtype)