import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
from captum.attr import DeepLift
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

class NN_model_interpreter:
    def __init__(self, model):
        self.model = model
        
    def integrated_grad(self, contour, baselines=-1000):
        ig = IntegratedGradients(self.model)
        gradients, error = ig.attribute(contour, baselines=baselines,
                                        return_convergence_delta=True)
        return gradients
    
    def deeplift(self, contour, baselines=-1000):
        dl = DeepLift(self.model)
        gradients, error = dl.attribute(contour, baselines=baselines,
                                        return_convergence_delta=True)
        return gradients
    
    def plot_heatmap(self, features, mean_grad, std_grad, plot_error=True):
        from collections import OrderedDict
        fig, ax = plt.subplots(figsize=(20,20))
        feat = features
        a = list(zip(mean_grad, std_grad))
        feat_dict = dict(zip(feat,a))
        sort_dict = OrderedDict(sorted(feat_dict.items(), key=lambda t: t[1]))   #t[1] mean sort by column 1
        sort_feat = list(sort_dict)
        sort_mean_std = np.array([feat_dict[item] for item in list(sort_dict)])
        y_pos = np.arange(len(sort_feat))
        
        feat_trans = {'WOMAC_TOTAL_SCORE':'WOMAC Total Score', 'WOMAC_DISABILITY_SCORE': 'WOMAC Disability Score',
                          'WOMAC_PAIN_SCORE': 'WOMAC Pain Score', 'WOMAC_STIFFNESS_SCORE': 'WOMAC Stiffness Score',
                          'AGE': 'Age', 'SEX': 'Sex', 'SEX_0': 'Sex: Male', 'SEX_1': 'Sex: Female', 'RACE': 'Race', 'RACE_0': 'Race:Others', 
                          'RACE_1': 'Race:White', 'RACE_2': 'Race:Black','RACE_3': 'Race:Asian',
                          'INJURY': 'History of Injury', 'SURGERY': 'History of Surgery', 'HEIGHT': 'Height', 
                          'SMOKING': 'Smoking Habit', 'BEER': 'Beer Intake Habit', 'MILK': 'Milk Intake Habit', 'ALCOHOL': 'Alcohol Intake Habit',
                          'DIAS': 'Diastolic Blood Pressure', 'SYS': 'Systolic Blood Pressure', 'MEAN_ARTERIAL_PRESSURE': 'Mean ArteriLal Pressure',
                          'BMI': 'BMI', 'DIABETES': 'Diabetes', 'WEIGHT': 'Weight', 
                          'KL_GRADE': 'KL-Grade', 'KL_GRADE_0.0': 'KL-Grade 0', 'KL_GRADE_1': 'KL-Grade 1',
                          'KL_GRADE_2': 'KL-Grade 2', 'KL_GRADE_3': 'KL-Grade 3', 'JSW': 'JSW', 
                          'OARSI_JSN_L': 'OARSI Joint Space Narrowing (Lateral)', 'OARSI_JSN_M': 'OARSI OARSI Joint Space Narrowing (Medial)',
                          'OARSI_OST_L': 'OARSI Osteophyte (Lateral)', 'OARSI_OST_M': 'OARSI Osteophyte (Medial)'
                          }
        
        sort_feat_trans = [feat_trans[i] for i in sort_feat]
        
        self.interp_dict_df = dict(zip(sort_feat_trans, sort_mean_std[:,0]))
        
        feat_color = {
             'firebrick': ['WOMAC_TOTAL_SCORE', 'WOMAC_DISABILITY_SCORE', 'WOMAC_PAIN_SCORE', 'WOMAC_STIFFNESS_SCORE'],
             'darkgrey': ['AGE', 'SEX', 'SEX_0', 'SEX_1', 'HEIGHT','RACE', 'RACE_0', 'RACE_1', 'RACE_2', 'RACE_3', 'WEIGHT'],
             'fuchsia': ['INJURY', 'SURGERY'],
             'royalblue': ['SMOKING', 'BEER', 'MILK', 'ALCOHOL'],
             'seagreen': ['DIAS', 'SYS', 'MEAN_ARTERIAL_PRESSURE', 'BMI', 'DIABETES'],
             'goldenrod': ['KL_GRADE', 'KL_GRADE_0', 'KL_GRADE_1', 'KL_GRADE_2', 'KL_GRADE_3', 'JSW', 'OARSI_JSN_L', 'OARSI_JSN_M', 'OARSI_OST_L', 'OARSI_OST_M']
                     }
        
        color_palette = []
        for item in sort_feat:
            color = list(feat_color.keys())[[item in list(feat_color.values())[i] for i in range(len(feat_color))].index(True)]
            color_palette.append(color)
        
        if plot_error:
            ax.barh(y_pos, sort_mean_std[:,0], xerr=sort_mean_std[:,1], align='center', color=color_palette)
        
        else:
            ax.barh(y_pos, sort_mean_std[:,0], align='center', color=color_palette)
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sort_feat_trans)
        ax.set_xlabel('Gradients')
        ax.set_title('MLP Interpretation')
        
        plt.show()
