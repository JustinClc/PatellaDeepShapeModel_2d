from pyefd import elliptic_fourier_descriptors, plot_efd
from tqdm import tqdm

import cv2
from skimage.feature import hog

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

def shape_feat_extract(shape_data, shape_models=list, efd_order=20, efd_plot=False):
    X_shape_des = []
    
    def efd_feat():
        cnt = shape_data[ID]
        coefs = elliptic_fourier_descriptors(cnt, order=efd_order, normalize=True)
        if efd_plot == True:
            plt.figure(figsize=(24, 8), dpi=80)
            plot_efd(coefs)
        efd_coefs = coefs.flatten()[3:]
        return efd_coefs
    
    def hu_feat():
        # Step:
        # 1. Define an image with 0 intensity
        # 2. Reconstruct a image mask of patella from the contour coordinates
        # 3. Calculate the Hu Moment from the mask
        cnt = deepcopy(shape_data[ID])
        cnt = cnt*100
        cnt = cnt.astype(int)
        img = np.zeros((1500,1500))                           # Define orginal image with 0 intensity
        cv2.drawContours(img, [cnt+700], -1, (255), cv2.FILLED) # construct mask from contour
        img_r = cv2.resize(np.uint8(img), (int(img.shape[0]/3), int(img.shape[1]/3)), # Reduce the image size
                           interpolation = cv2.INTER_AREA)
        hu_moments = cv2.HuMoments(cv2.moments(img_r)).flatten()
        return hu_moments
    
    def hog_feat():
        # Step:
        # 1. Define an image with 0 intensity
        # 2. Reconstruct a image mask of patella from the contour coordinates
        # 3. Calculate the HoG features from the mask
        cnt = deepcopy(shape_data[ID])
        cnt = cnt * 100
        cnt = cnt.astype(int)
        #x = cnt[:, 1]
        #y = cnt[:, 0]
        #plt.scatter(x, y)
        #plt.show()
        img = np.zeros((1500,1500))                           # Define orginal image with 0 intensity
        cv2.drawContours(img, [cnt], -1, (255), cv2.FILLED) # construct mask from contour
        img_r = cv2.resize(np.uint8(img), (int(img.shape[0]/3), int(img.shape[1]/3)), # Reduce the image size
                           interpolation = cv2.INTER_AREA)
        fd = hog(img_r, orientations=8, pixels_per_cell=(250, 250),
                 cells_per_block=(10, 10), visualize=False,
                 feature_vector=True)
        return fd

    for ID in tqdm(shape_data):
        # Stack all tda representations horizontally to form a multi-tda vector
        features = []
        for model in shape_models:
            if model == 'efd':
                feat = efd_feat()
            elif model == 'hu_moment':
                feat = hu_feat()
            elif model == 'hog':
                feat = hog_feat()
            features.append(feat)
            
        multi_feat = np.hstack(features)
        X_shape_des.append(multi_feat)
    X_shape_des = np.array(X_shape_des)
    X_shape_des[np.isnan(X_shape_des)] = 0
    
    return X_shape_des