import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))
import torch

DATA_PATH = "./data"
BATCH_SIZE = 4
NUM_WORKERS = 2
NORMALIZE = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 5
MODEL_PATH = "./models"
PIXELS = (32, 32)