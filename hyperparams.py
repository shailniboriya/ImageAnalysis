from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import os

# set hyperparameters
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256,256))
image_size = 0
#please change the location of  input folder according to your machine
curr_dir= os.getcwd()
directory_root = "./input/train/"
out_dir = "./output/"
width=256
height=256
depth=3

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
