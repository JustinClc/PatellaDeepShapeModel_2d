from tqdm import tqdm
import cv2
import numpy as np

from pyts.transformation import ShapeletTransform, ROCKET
from pyts.classification import SAXVSM
from pyts.bag_of_words import BagOfWords

from ..transforms import *

def shaplet(shape_data, shape_models=list, efd_order=20, efd_plot=False):
    
    
    return X_shape_des