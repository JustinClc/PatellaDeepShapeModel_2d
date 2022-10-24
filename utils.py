import torch
import torch.nn.functional as F

import 

########################################################################################
#Model training utilities
########################################################################################

def freeze_modules(model, module):
    for name, p in model.named_parameters():
        if module in name:
            p.requires_grad = False

def unfreeze_modules(model, module='all'):
    if module == 'all':
        for p in model.parameters():
            p.requires_grad = True
    else:
        for name, p in model.named_parameters():
            if module in name:
                p.requires_grad = True
