import torch
from torch import nn
import torch.nn.functional as F

class FeatureExtractor:
    def __init__(self,
                 model,
                 target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = list()

    def save_grad(self, grad):
        self.gradients.append(grad)

    def get_grad(self):
        return self.gradients

    def __call__(self):
        target_activations = list()

