#
print('Loading SG2 utilis,', end='')
from ss_utils.shuang_utils import save_variable, load_variavle
from ss_utils.SG_utils import *

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
model_dic = dict(original_HUMAN=['network_human', 1024, 'HUMAN'],

                 original_FFHQ=['network', 1024, 'FFHQ'],
                 original_AFHQ=['network', 512, 'AFHQ'],
                 original_Metface=['network', 1024, 'Metface'],

                 modify_FFHQ=['modify_networks', 1024, 'FFHQ'],
                 modify_AFHQ=['modify_networks', 512, 'AFHQ'],
                 modify_Metface=['modify_networks', 1024, 'Metface'],

                 feature_FFHQ=['feature_network', 1024, 'FFHQ'],
                 feature_AFHQ=['feature_network', 512, 'AFHQ'],
                 feature_Metface=['feature_network', 1024, 'Metface'],

                 ablation=['ablation', 1024, 'FFHQ'],

                 minor_FFHQ=['minor_feature', 1024, 'FFHQ'],
                 minor_AFHQ=['minor_feature', 512, 'AFHQ'],
                 minor_Metface=['minor_feature', 1024, 'Metface'],
                 microphone_FFHQ=['microphone', 1024, 'FFHQ'],

                 ethics_feature=['ethics_feature', 1024, 'FFHQ'],
                 ethics_inversion=['ethics_inversion', 1024, 'FFHQ'],

                 ethics_FFHQ=['ethics_feature', 1024, 'FFHQ'],
                 ethics_AFHQ=['ethics_feature', 512, 'AFHQ'],
                 ethics_Metface=['ethics_feature', 1024, 'Metface'],

                 CIFAR10=['network', 32, 'CIFAR10'])
# Chosed_model='ethics_feature'
# Chosed_model = 'ethics_inversion'
# set modify_index
FFHQ_modify_index = {
    'b0_conv1': 1, 'b1_conv0': 1, 'b2_conv0': 1, 'b3_conv0': 1, 'b4_conv0': 0, 'b5_conv0': 0, 'b6_conv0': 0,
    'b7_conv0': 0, 'b8_conv0': 0, 'b9_conv0': 0,
    'b1_conv1': 1, 'b2_conv1': 1, 'b3_conv1': 1, 'b4_conv1': 0, 'b5_conv1': 0, 'b6_conv1': 0, 'b7_conv1': 0,
    'b8_conv1': 0, 'b9_conv1': 0,
}
save_variable(FFHQ_modify_index, 'FFHQ_modify_index.txt')


def SG2_record_variation(seed, G_copy=None):
    Key_all = {'modify': False, 'modify_0': False, 'modify_1': False, 'record_variation': True, 'heat_display': False,
               'modify_num': False, 'modi_index_all': False, 'ratio_all': True, 'x_modi_yet': False, 'x_all': False}
    save_variable(Key_all, 'Key_all.kpl')
    if G_copy == None:
        Chosed_model = 'ethics_feature'
        G_copy, import_model, img_resolution, dataset = model_select(model_dic, Chosed_model)
    label = torch.zeros([1, G_copy.c_dim], device=device)
    truncation_psi = 1
    noise_mode = 'const'
    heat_display = 0
    z = compute_z(seed, device)
    img, dic_para = G_copy(z, label, dataset, heat_display, truncation_psi=truncation_psi, noise_mode=noise_mode)
    #     ratio_all_dic[str(seed)]=dic_para['ratio_all']
    # save_variable(dic_para['ratio_all'],'ratio_all_dic/'+str(seed)+'_ratio_all_dic.kpl')
    return img, dic_para

def SG2_Generation( Chosed_model, seed, operation='Original', truncation_psi = 1,G_copy=None, **kwargs):
    Key_all = {'modify': False, 'modify_0': False, 'modify_1': False, 'record_variation': False, 'heat_display': False,
               'modify_num': False, 'modi_index_all': False, 'ratio_all': False, 'x_modi_yet': False, 'x_all': False}
    if operation == 'modify':
        Key_all['modify'] = True
        Key_all['modify_0'] = True
        Key_all['modify_0'] = True
    if operation == 'feature':
        Key_all['record_variation'] = True
    if kwargs:
        for i in kwargs:
            Key_all[str(i)] = True

    save_variable(Key_all, 'Key_all.kpl')
    if G_copy == None:
        # Chosed_model = 'ethics_feature'
        G_copy, import_model, img_resolution, dataset = model_select(model_dic, Chosed_model)
    label = torch.zeros([1, G_copy.c_dim], device=device)
    noise_mode = 'const'
    heat_display = 0
    z = compute_z(seed, device)
    img, dic_para = G_copy(z, label, dataset, heat_display, truncation_psi=truncation_psi, noise_mode=noise_mode)
    #     ratio_all_dic[str(seed)]=dic_para['ratio_all']
    # save_variable(dic_para['ratio_all'],'ratio_all_dic/'+str(seed)+'_ratio_all_dic.kpl')
    return img, dic_para

print('    Complete!!!')
