#
print('Loading SG3 utilis,', end='')

from ss_utils.shuang_utils import save_variable, load_variavle
from ss_utils.SG_utils import *

import pickle
import seaborn as sns
import os

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
model_dic = dict(original_LHQ=['networks_stylegan3',256,'LHQ_256'],
                 feature_LHQ=['feature_stylegan3',256,'LHQ_256'],
                 modify_LHQ=['modi_LHQ_SG3 ',256,'LHQ_256'],

                 original_BENCH=['networks_stylegan3',512,'BENCH'],
                 feature_BENCH=['feature_stylegan3',512,'BENCH'],
                 modify_BENCH=['modify_stylegan3',512,'BENCH'],
                 
                 original_WIKI=['networks_stylegan3',1024,'WIKI'],
                 feature_WIKI=['feature_stylegan3',1024,'WIKI'],
                 modify_WIKI=['modi_WIKI_SG3',1024,'WIKI'],


                 original_FFHQ=['networks_stylegan3', 1024, 'stylegan3_ffhq'],
                 original_AFHQ=['networks_stylegan3', 512, 'stylegan3_afhq'],
                 original_Metface=['networks_stylegan3', 1024, 'Metface'],

                 modify_FFHQ=['modify_stylegan3', 1024, 'stylegan3_ffhq'],
                 modify_AFHQ=['modify_stylegan3', 512, 'stylegan3_afhq'],
                 modify_Metface=['modify_stylegan3', 1024, 'Metface'],

                 feature_FFHQ=['feature_stylegan3', 1024, 'stylegan3_ffhq'],
                 feature_AFHQ=['feature_stylegan3', 512, 'stylegan3_afhq'],
                 feature_Metface=['feature_stylegan3', 1024, 'Metface'],

                 ablation=['ablation', 1024, 'FFHQ'],

                 minor_FFHQ=['minor_feature', 1024, 'stylegan3_ffhq'],
                 minor_AFHQ=['minor_feature', 512, 'stylegan3_afhq'],
                 minor_Metface=['minor_feature', 1024, 'Metface'],
                 microphone_FFHQ=['microphone', 1024, 'stylegan3_ffhq'],

                 ethics_feature=['ethics_feature', 1024, 'stylegan3_ffhq'],
                 ethics_inversion=['ethics_inversion', 1024, 'stylegan3_ffhq'],

                 CIFAR10=['network', 32, 'CIFAR10'])
# print('Loading SG3 model')
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

def SG3_generate(Chosed_model, seed, truncation_psi = 1,G_copy=None):
    if G_copy == None:
        G_copy, import_model, img_resolution, dataset = model_select(model_dic, Chosed_model)
    z = compute_z(seed, device)
    label = torch.zeros([1, G_copy.c_dim], device=device)
    noise_mode = 'const'
    if Chosed_model[:7] == 'feature':
        img, x_mean = G_copy(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        return img, x_mean
    else:
        img = G_copy(z, label, truncation_psi=truncation_psi,noise_mode=noise_mode)
        x_mean=[]
        return img, x_mean


print('    Complete!!!')
