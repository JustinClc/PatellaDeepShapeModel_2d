import numpy as np
from csv import writer
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, classification_report, confusion_matrix 

import sys
sys.path.append(r'D:/OneDrive - The Hong Kong Polytechnic University/文件/KOA/code/patella_deep_shape_analysis/models')
sys.path.append(r'D:/OneDrive - The Hong Kong Polytechnic University/文件/KOA/code/patella_deep_shape_analysis\data_utils')
from loss_func import *
from models.CircularFCN import CircularFCN
from models.CircularConvSelfAttNet import SelfAttCirCNN

class DeepShapeAnalysis(pl.LightningModule):
    def __init__(self,
                 task,
                 self_att,
                 loss,
                 lr,
                 loss_alpha):
        super().__init__()
        """
        :param: 
        arch: architecture -- str 'circularFCN' or 'circularSACNN'
        self_att: employed self-attention or not -- binary
        loss: loss function -- str 'BCE', 'focal'
        lr: learning rate -- float
        loss_alpha: alpha parameter for focal loss -- float [0,1]
        """
        #self.arch = arch
        self.task = task
        self.loss = loss
        self.self_att = self_att
        self.lr = lr
        self.loss_alpha = loss_alpha

        if not self_att:
            self.arch = 'CirFCN'
            self.model = CircularFCN(hidden_layer=[128,256,128], kernel_size=[7,5,3])
        else:
            self.arch = 'SACirCNN'
            self.model = SelfAttCirCNN()

        assert(loss in ['bce', 'focal'])
        if loss == 'bce':
            self.loss_fc = BCELoss()
        elif loss == 'focal':
            self.loss_fc = BinaryFocalLoss(weight='mean')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fc(y_pred, y)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.detach()
        y_pred = self.model(x)
        loss = self.loss_fc(y_pred, y.float()).detach()
        self.log('valid_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(),
                                  lr=self.lr,
                                  weight_decay=0.01,
                                  amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optim,
                                                      base_lr=self.lr,
                                                      max_lr=self.lr*100,
                                                      cycle_momentum=False,
                                                      mode="triangular2")
        #scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.1)
        return [optim], [scheduler]

    def save(self, fname):
        path = r'./trained_models'
        file_wo = fname + '_weight_only.pth'
        torch.save(self.model.state_dict(), os.path.join(path, file_wo))
        file = fname + '.pth'
        torch.save(self.model, os.path.join(path, file))

    def load(self, fname):
        path = r'./trained_models'
        file = fname + '_weight_only.pth'
        self.model.load_state_dict(torch.load(os.path.join(path, file)), strict=False)
        self.model.to("cuda:0")
        return self

    def test(self, dloader, model_name, threshold=0.5, save=True):
        """
        Parameters
        ----------
        model_name
        threshold: decision threshold, default=0.5
        save: save the logs to the csv file?
        """
        y_probs = np.array([])
        y_preds = np.array([])
        y_trues = np.array([])

        iteration = 0
        progress = tqdm(total=dloader.__len__())
        self.model.to('cuda:0')
        self.model.eval()
        for img, label in dloader:
            progress.update(1)
            iteration += 1
            img = img.to('cuda:0')
            y_prob = self.model(img)
            y_pred = y_prob > threshold
            y_prob = y_prob.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
            y_true = label.cpu().detach().numpy()

            y_probs = np.concatenate((y_probs, y_prob))
            y_preds = np.concatenate((y_preds, y_pred))
            y_trues = np.concatenate((y_trues, y_true))

        acc = accuracy_score(y_trues, y_preds)
        pr = precision_score(y_trues, y_preds)
        re = recall_score(y_trues, y_preds)
        f1 = f1_score(y_trues, y_preds)
        auc = roc_auc_score(y_trues, y_probs)

        cm = confusion_matrix(y_trues, y_preds)
        TN = cm[0,0]
        TP = cm[1,1]
        FN = cm[1,0]
        FP = cm[0,1]

        date_time = datetime.datetime.now()

        target_names = ['Negative', 'Positive']
        print(classification_report(y_trues, y_preds, target_names=target_names))
        print('AUROC score:', auc)
        print(confusion_matrix(y_trues, y_preds, labels=list(range(len(target_names))), normalize=None))

        info = [date_time, model_name,
                self.loss, self.lr,
                self.self_att,
                acc, pr, re, f1, auc, TN, TP, FN, FP]

        fname = 'testing_performance_'+self.task+'.csv'
        if os.path.isfile(fname):
            with open(fname, 'a', newline='') as f:
                writer_object = writer(f)
                writer_object.writerow(info)
                f.close()
        else:
            df_performance = pd.DataFrame(data=np.array([info]),
                                          columns=['Date-time', 'Model name',
                                                   'Loss func', 'lr',
                                                   'Self-att', 'Accuracy',
                                                   'Precision', 'Recall', 'F1',
                                                   'AUC', 'TN', 'TP', 'FN', 'FP'])
            df_performance.to_csv(fname, index=False)

        return y_preds, y_trues
