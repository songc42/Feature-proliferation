#
print('Loading SG2 utilis,', end='')
from ss_utils.shuang_utils import save_variable, load_variavle
from ss_utils.SG_utils import model_select, compute_z

import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from typing import List, Optional
import matplotlib.pyplot as plt
import click
import dnnlib
import numpy as np
import PIL.Image
import torch

device = torch.device('cuda')
# device=torch.device('cpu')

# Model select
model_dic = dict(

                 original_FFHQ=['network', 1024, 'FFHQ'],
                 original_AFHQ=['network', 512, 'AFHQ'],
                 original_Metface=['network', 1024, 'Metface'],

                 modify_FFHQ=['modify_networks', 1024, 'FFHQ'],
                 modify_AFHQ=['modify_networks', 512, 'AFHQ'],
                 modify_Metface=['modify_networks', 1024, 'Metface']

)

# Specify modify_index
Modify_index = {
    'b0_conv1': 1, 'b1_conv0': 1, 'b2_conv0': 1, 'b3_conv0': 1, 'b4_conv0': 0, 'b5_conv0': 0, 'b6_conv0': 0,
    'b7_conv0': 0, 'b8_conv0': 0, 'b9_conv0': 0,
    'b1_conv1': 1, 'b2_conv1': 1, 'b3_conv1': 1, 'b4_conv1': 0, 'b5_conv1': 0, 'b6_conv1': 0, 'b7_conv1': 0,
    'b8_conv1': 0, 'b9_conv1': 0,
}
save_variable(Modify_index, 'Modify_index.txt')




def SG2_Generation( Chosed_model, seed, operation='SG2', truncation_psi = 1,G_copy=None, **kwargs):

    # If modify, Project seed to Z
    z = compute_z(seed, device)
    Key_all = {}
    if operation == 'modify_SG2':
        Key_all['modify'] = True
    save_variable(Key_all, 'Key_all.kpl')
    
    # Load generator
    if G_copy == None:
        G_copy, import_model, img_resolution, dataset = model_select(model_dic, Chosed_model)
        
    # Generate image
    label = torch.zeros([1, G_copy.c_dim], device=device)
    noise_mode = 'const'  
    img, dic_para = G_copy(z, label, dataset, truncation_psi=truncation_psi, noise_mode=noise_mode)

    return img, dic_para

print('    Complete!!!')
