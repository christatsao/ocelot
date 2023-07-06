import torch

MY_DEVICE = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = True if MY_DEVICE == 'cuda' else False
D_TYPE = torch.float32

DATA_DIR = '../../ocelot2023_v0.1.2/'
META_PATH = DATA_DIR + 'metadata.json'

IMG_DIR = DATA_DIR + 'images/'
IMG_TRAIN_DIR = IMG_DIR + 'train/'
IMG_TRAIN_CELL_DIR = IMG_TRAIN_DIR + 'cell/'
IMG_TRAIN_TISS_DIR = IMG_TRAIN_DIR + 'tissue/'

ANN_DIR = DATA_DIR + 'annotations/'
ANN_TRAIN_DIR = ANN_DIR + "train/"
ANN_TRAIN_CELL_DIR = ANN_TRAIN_DIR + "cell/"
ANN_TRAIN_TISS_DIR = ANN_TRAIN_DIR + "tissue/"

DATA_PATHS = [IMG_TRAIN_CELL_DIR, ANN_TRAIN_CELL_DIR, IMG_TRAIN_TISS_DIR, ANN_TRAIN_TISS_DIR, META_PATH]
