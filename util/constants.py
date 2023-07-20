import torch
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT,'Ocelot' ,'data','ocelot2023_v0.1.2')
META_PATH = os.path.join(DATA_DIR,'metadata.json')

IMG_DIR = os.path.join(DATA_DIR,'images')
IMG_TRAIN_DIR = os.path.join(IMG_DIR,'train')
IMG_TRAIN_CELL_DIR = os.path.join(IMG_TRAIN_DIR, 'cell')
IMG_TRAIN_TISS_DIR = os.path.join(IMG_TRAIN_DIR,'tissue')

ANN_DIR = os.path.join(DATA_DIR ,'annotations')
ANN_TRAIN_DIR = os.path.join(ANN_DIR , "train")
ANN_TRAIN_CELL_DIR = os.path.join(ANN_TRAIN_DIR , "cell")
ANN_TRAIN_TISS_DIR = os.path.join(ANN_TRAIN_DIR ,"tissue")

DATA_PATHS = [IMG_TRAIN_CELL_DIR, ANN_TRAIN_CELL_DIR, IMG_TRAIN_TISS_DIR, ANN_TRAIN_TISS_DIR, META_PATH]

SAMPLE_SHAPE = (1024, 1024, 3)
