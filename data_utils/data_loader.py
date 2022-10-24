import cv2 as cv
from copy import deepcopy
import numpy as np
from scipy import ndimage
import pandas as pd
import glob
from ipywidgets import IntProgress
from operator import itemgetter
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from procrustes_analysis import *

import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule

def reduce_contour(contour, target_num_pt=60):
    num_pt = len(contour)
    spacing = num_pt/target_num_pt
    contour_red = []
    for i in range(target_num_pt):
        # Find idex
        idx = i*spacing
            
        # Interpolate coordinates
        dec_part, int_part = math.modf(idx)
            
        if int_part+1 <= num_pt-1:
            low_lim, up_lim = int(int_part), int(int_part)+1
        elif (int_part <= num_pt-1) and (int_part+1 > num_pt-1):
            low_lim, up_lim = int(int_part), int(int_part)+1-num_pt
        elif int_part > num_pt-1:
            low_lim, up_lim = int(int_part)-num_pt, int(int_part)+1-num_pt
            
        low_port, up_port = dec_part, 1-dec_part
            
        [x_l, y_l] = contour[low_lim]
        [x_u, y_u] = contour[up_lim]
        x_n = x_l*low_port + x_u*up_port
        y_n = y_l*low_port + y_u*up_port
        #contour_red.append([np.around(x_n, 0).astype(int), np.around(y_n, 0).astype(int)])
        contour_red.append([x_n, y_n])
    return contour_red

def plot_contours(cont_list, mean_cont=None, plot_centroids=False):
    # cont_list contains list of contours
    img = np.ones((700,700,3), np.uint8)*255

    if plot_centroids == True:
        colors = [(255, 50, 50), (50, 255, 50), (50, 50, 255)]
        thickness1 = 2
        for i, cont in enumerate(cont_list):
            cv.polylines(img, [cont], 
                        True, colors[i], thickness1) 
    else:
        color1 = (255, 50, 50)
        thickness1 = 1
        for cont in cont_list:
            cv.polylines(img, [cont], 
                        True, color1, thickness1) 
        
    if type(mean_cont)==np.ndarray or list:
        color2 = (0, 255, 255)
        thickness2 = 1
        cv.polylines(img, [mean_cont], 
                        True, color2, thickness2)
    
    resized = cv.resize(img, None, fx=1, fy=1, interpolation=cv.INTER_AREA)    

    cv.imshow('My Image', resized)
    cv.waitKey(0)
    cv.destroyAllWindows()

def plot_single_contour(cont):
    # cont_list contains list of contours
    img = np.ones((700,700,3), np.uint8)*255

    color = (100, 100, 100)
    thickness = 2
    cv.polylines(img, [cont], 
                True, color, thickness)
    
    resized = cv.resize(img, None, fx=1, fy=1, interpolation=cv.INTER_AREA)    

    cv.imshow('My Image', resized)
    cv.waitKey(0)
    cv.destroyAllWindows()

def contour_1D_to_2D(contour_list, transl=0):
    sample_size = len(contour_list)
    cont_length = contour_list[0].shape[0]
    return (np.array(contour_list)+transl).reshape(sample_size, int(cont_length/2), 2).astype(np.int32)

def shape_normalize(contour, scale=100): # Normalisation of input contour, minmax
    contour_max = np.max(contour, axis=0)
    contour_min = np.min(contour, axis=0)
    contour_diff = contour_max - contour_min

    # Min-max normalisation
    contour_norm = (contour - contour_min)/contour_diff
    contour_norm[:,0] = contour_norm[:,0]*scale
    contour_norm[:,1] = contour_norm[:,1]*(scale*contour_diff[1]/contour_diff[0])

    # Centering
    contour_mean = np.mean(contour_norm, axis=0)
    contour_center = contour_norm - contour_mean
    return contour_center

class load_data:
    def __init__(self, 
                 img_directory, 
                 img_exclude_csv,
                 clinical_df):
        self.DIRECTORY = img_directory
        self.img_exclude_csv = os.path.join(self.DIRECTORY, img_exclude_csv)
        self.DF_FNAME = clinical_df
        
    def get_landmark(self, img_path, normalize=True):
        img = cv.imread(img_path)#, cv.IMREAD_GRAYSCALE)
        imgrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgrey,200,255,cv.THRESH_BINARY)
        #thresh = ndimage.binary_fill_holes(thresh).astype(np.uint8) #fill up holes in the binary mask

        cnts, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        #contours = measure.find_contours(imgrey, 0.8)

        # Select the suitable contour
        for c in cnts:
            area = cv.contourArea(c)
            if area > 100 and area < 65000: # <-- this threshold is defined by the area of the rectangular frame, must be smaller that it
                #v.drawContours(img, [c], -1, (0,0,0), -1)
                #print(c.shape)
                cnt = c.reshape(-1,2)   
        return img, cnt
    
    def draw_contour(self, img_id, img, cnt):
        for (x,y) in cnt:
            cv.circle(img, (x, y), 1, (255, 0, 0), 3)
        save_path = os.path.join('./contour', img_id+'.png')
        cv.imwrite(save_path, img)
        
    def get_all_landmarks(self):
        EXT = 'V0 (Baseline visit)/*.png'
        PATHS = os.path.join(self.DIRECTORY, EXT)
        img_exclude_list = pd.read_csv(self.img_exclude_csv)
        img_exclude_list['Bad mask'] = img_exclude_list['Bad mask'].astype(str) + '.png'

        contour_list = []
        id_list = []
        for path in tqdm(glob.glob(PATHS)):
            if path in img_exclude_list['Bad mask'].values: #skipping the image files on the exclude list
                print('Skipped ', path)
                continue
            else:
                image, contour = self.get_landmark(path)
                contour_list.append(np.array(contour))
                img_id = path.split('\\')[-1]
                img_id = img_id.split('.')[0]
                id_list.append(img_id)
                #self.draw_contour(img_id, image, contour)
        
        return dict(zip(id_list, contour_list))
    
    def red_all_contour(self, contour_list, target_num_pt, normalize=True, transpose=True):
        """
        Args:
            contour_list: list containing contour point sets
            target_num_pt: number of points sampled from the original point set
            normalize: minmax and centering normalization of point set
            transpose: if True, transpose the pointset in order to attain time series-like form

        Returns:
            dict containing index and contour with sampled point set
        """
        contour_red_list = []
        for i, c in enumerate(contour_list.values()):
            contour_red = reduce_contour(c, target_num_pt=target_num_pt)
            if normalize == True:
                contour_red = shape_normalize(contour_red, scale=100)
            if transpose == True:
                contour_red = np.transpose(contour_red)
            contour_red_list.append(contour_red)
        return dict(zip(list(contour_list), contour_red_list))
    
    def gen_procruste_analysis(self, contour_list):
        # Flatten the contours [n_pt x 2] for the generalised procrustes analysis
        contours = [c.reshape(-1) for c in list(contour_list.values())]
        mean_shape, alligned_shape_list = generalized_procrustes_analysis(contours)
        alligned_shape_list = [contour.reshape(int(len(contour) / 2), 2) for contour in alligned_shape_list]
        id_list = list(contour_list)
        return mean_shape, dict(zip(id_list, alligned_shape_list))

    def load_target(self, alligned_shape_list, traintest_split=True, random_seed=42, target='PFOA'):
        """
        Args:
            alligned_shape_list:
            traintest_split: Binary
            target: labels, i.e. PFOA, TFOA, whole joint OA, KL-grade, KR_visit (KR_in_84m)

        Returns:
            if train_test_split is True: return splited contour dict (with index) and targets
            if train_test_split is False: return all contour dict (with index) and targets
        """
        df = pd.read_csv(self.DF_FNAME)
        df = df[df['PatientID'].isin(list(alligned_shape_list))]

        drop_idx = df[df[target].isna()].index
        df.drop(drop_idx, axis=0, inplace=True)
        
        all_id = df['PatientID']
        if target == 'KR_visit':
            all_targets = df['KR_visit'] >= 0
        else:
            all_targets = df[target]

        if traintest_split == True:
            train_id, test_id, y_train, y_test = train_test_split(all_id, all_targets, test_size=0.2, random_state=random_seed)
            train_shapes = dict(zip(train_id, itemgetter(*train_id)(alligned_shape_list)))
            test_shapes = dict(zip(test_id, itemgetter(*test_id)(alligned_shape_list)))
            return train_shapes, test_shapes, y_train, y_test
        else:
            all_shapes = dict(zip(all_id, itemgetter(*all_id)(alligned_shape_list)))
            return all_shape, all_targets

def GetTrainTransforms():
    return transforms.Compose([transforms.ToTensor()])
                               #tg.transforms.RandomRotate(degrees=10),
                               #tg.transforms.RandomFlip(axis=0, p=0.3)])

def GetValTestTransforms():
    return transforms.Compose([transforms.ToTensor()])

def weighted_data_sampler(data_label):
    data_label = data_label.astype(int)
    # Weighted sampler for imbalancd training
    unique_labels, counts = np.unique(data_label, return_counts=True)
    # Calculate weight for each class
    class_weights = [sum(counts)/c for c in counts]
    # Assign weight to each sample in the dataset
    sample_weights = [class_weights[e] for e in data_label]
    sampler = WeightedRandomSampler(sample_weights, len(data_label), replacement=True)
    return sampler

class ContourDataset(Dataset):
    def __init__(self, contour_dict, target_list, trsf):
        self.transform = trsf
        self.target_list = np.array(target_list)
        print('========================',self.target_list.shape)
        self.contour_list = list(contour_dict.values())

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, idx):
        contour = self.contour_list[idx]
        contour = self.transform(contour)
        target = self.target_list[idx]
        target = torch.tensor(int(target)).to(torch.float)
        return contour, target

class ContourDataModule(LightningDataModule):
    def __init__(self,
                 prediction_target,
                 target_num_pt,
                 mask_dir=r'F:/MOST/MOST-Images (Radiograph & MRI)/MOST Knee Radiographs/inference/AttResNet18-Unet_crop',
                 img_exclude_csv='mask_exclude_list.csv',
                 df_fname=r'./data/MOST_0m_1500_clinical_data.csv',
                 val_ratio=0.2,
                 random_state=40,
                 weighted_sampling=True,
                 batch_size=32) -> None:
        super().__init__()
        
        self.batch_size = batch_size

        EC = load_data(img_directory=mask_dir, 
                       img_exclude_csv=img_exclude_csv,
                       clinical_df=df_fname)
        contour_list = EC.get_all_landmarks()
        red_contour_list = EC.red_all_contour(contour_list, target_num_pt=target_num_pt, normalize=True)
        self.train_shapes, self.valid_shapes, self.y_train, self.y_valid = EC.load_target(red_contour_list, 
                                                                                          traintest_split=True, 
                                                                                          random_seed=random_state,
                                                                                          target=prediction_target)
        self.train_trsf = GetTrainTransforms()
        self.valid_trsf = GetValTestTransforms()

        self.sampler = weighted_data_sampler(self.y_train) if weighted_sampling else None

        self.train_dt = ContourDataset(self.train_shapes, self.y_train, self.train_trsf)
        self.valid_dt = ContourDataset(self.valid_shapes, self.y_valid, self.valid_trsf)

    def setup(self, stage=None):
        self.train_dt = ContourDataset(self.train_shapes, self.y_train, self.train_trsf)
        self.valid_dt = ContourDataset(self.valid_shapes, self.y_valid, self.valid_trsf)

    def train_dataloader(self):
        return DataLoader(self.train_dt, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          sampler=self.sampler, 
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dt, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          pin_memory=True)

