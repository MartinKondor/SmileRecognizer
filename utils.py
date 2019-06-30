import os
from PIL import Image

from sklearn.datasets import get_data_home
import numpy as np


DATA_DIR = os.path.join(get_data_home(), 'lfw_home', 'lfw_funneled')
IMG_SHAPE = (90, 80)


def loadimg(file_list, i):
    img_filename = file_list[i]
    img_path = '_'.join(file_list[i].split('_')[:2])
    filename = os.path.join(DATA_DIR, img_path, img_filename)
    return vectorizeimg(filename)


def vectorizeimg(filename):
    return np.array(Image.open(filename).resize(IMG_SHAPE))
