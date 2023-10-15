print('Loading Ethics,', end='')

from ss_utils.shuang_utils import load_variavle, save_variable
from ss_utils.SG_utils import *
import torch

###------------------------------concatenate elements of all layers of all imgs and combine------------------------------------###
def CVPR_concate_layers_combine(dict_keys, base_file, seed_num=10000, zfill=True):  # used to be called sort_dic
    # Input : dict_keys : a list of name of layers, base_file : a string of name of loaded data(missing index).
    # dict_keys = ['b0_conv1', 'b1_conv0', 'b1_conv1', 'b2_conv0', 'b2_conv1', 'b3_conv0', 'b3_conv1']
    seed_base = 0
    # seed_num = 10000
    # ratio_vector_all=torch.tensor()
    for i in range(seed_num):
        print(i)
        seed = seed_base + i
        if zfill:
            file_name = base_file.format(str(seed).zfill(5))
        else:
            file_name = base_file.format(str(seed))
        element_all = load_variavle(file_name)
        for j in dict_keys:
            layer_element = element_all[j]
            if j == dict_keys[0]:
                element_vector = layer_element
            else:
                element_vector = torch.cat((element_vector, layer_element), dim=0)
        if i == 0:
            element_vector_all = element_vector.unsqueeze(0)
        else:
            element_vector_all = torch.cat((element_vector_all, element_vector.unsqueeze(0)), dim=0)
    return element_vector_all

###------------------------------concatenate elements of all layers of all imgs and combine------------------------------------###
def concate_layers_combine(dict_keys, base_file, seed_num=10000, zfill=True):  # used to be called sort_dic
    # Input : dict_keys : a list of name of layers, base_file : a string of name of loaded data(missing index).
    # dict_keys = ['b0_conv1', 'b1_conv0', 'b1_conv1', 'b2_conv0', 'b2_conv1', 'b3_conv0', 'b3_conv1']
    seed_base = 0
    # seed_num = 10000
    # ratio_vector_all=torch.tensor()
    for i in range(seed_num):
        print(i)
        seed = seed_base + i
        if zfill:
            file_name = base_file.format(str(seed).zfill(5))
        else:
            file_name = base_file.format(str(seed))
        element_all = load_variavle(file_name)
        for j in dict_keys:
            layer_element = element_all[j]
            if j == dict_keys[0]:
                element_vector = layer_element
            else:
                element_vector = torch.cat((element_vector, layer_element), dim=0)
        if i == 0:
            element_vector_all = element_vector.unsqueeze(0)
        else:
            element_vector_all = torch.cat((element_vector_all, element_vector.unsqueeze(0)), dim=0)
    return element_vector_all

###------------------------------concatenate elements of all layers of all imgs and save--------------------------------------###
def concate_layers_save(dict_keys, base_file, save_name, seed_num=10000, save_file=True, zfill=True,base_file_full=False,save_name_full=False):# used to be called sort_dic
    # Input : dict_keys : a list of name of layers, base_file : a string of name of loaded data(missing index), save_file : filename to save.
    # dict_keys = ['b0_conv1', 'b1_conv0', 'b1_conv1', 'b2_conv0', 'b2_conv1', 'b3_conv0', 'b3_conv1']
    seed_base = 0
    # ratio_vector_all=torch.tensor()
    if base_file_full:
        seed_num = 1
    for i in range(seed_num):
        # print(i)
        seed = seed_base+i
        if zfill:
            file_name = base_file.format(str(seed).zfill(5))
        else:
            file_name = base_file.format(str(seed))
        if base_file_full:
            file_name = base_file
        element_all = load_variavle(file_name)

        for j in dict_keys:
            layer_element = element_all[j]
            if j == dict_keys[0]:
                element_vector = layer_element
            else:
                element_vector = torch.cat((element_vector, layer_element), dim=0)
        #ratio_vector_all=torch.cat((ratio_vector.unsqueeze(0),ratio_vector_all),dim=1)S
        if save_file:
            if save_name_full:
                save_variable(element_vector, save_name)
            else:
                save_variable(element_vector, save_name.format(str(seed).zfill(5)))
    return element_vector

###------------------------------------------------concatenate 10000 vectors--------------------------------------------------###
def concate_vectors(base_file,seed_num=10000,zfill=True):# used to be called sort_dic
    # Input : dict_keys : a list of name of layers, base_file : a string of name of loaded data(missing index), save_file : filename to save.
    seed_base = 0
    # ratio_vector_all=torch.tensor()
    for i in range(seed_num):
        print(i)
        seed = seed_base + i
        if zfill:
            file_name = base_file.format(str(seed).zfill(5))
        else:
            file_name = base_file.format(str(seed))
        vector_one = load_variavle(file_name)
        if i == 0:
            vector_all = vector_one.unsqueeze(0)
        else:
            vector_all = torch.cat((vector_all, vector_one.unsqueeze(0)), dim=0)
    print('concatenate{}vectors'.format(seed_num))
    return vector_all

###---------------------------------------extract ratios of 10000 inverted images--------------------------------------------###
def extract_ratio(seed_num=10000):
    # Input : Number of images you want to extract
    # Output : Concatenated ratios in dim 0
    base = '/home/shuang/Jupyter_linux/StyleGAN2++/ratio_vector_10000/'
    seed_base = 0
    seed_num = seed_num
    # ratio_vector_all=torch.tensor()
    for i in range(seed_num):
        # print(i)
        seed = seed_base + i
        file_name = base + str(seed).zfill(5) + '_ratio_vector.kpl'
        ratio_vector = load_variavle(file_name)
        if i == 0:
            ratio_vector_all = ratio_vector.unsqueeze(0)
        else:
            ratio_vector_all = torch.cat((ratio_vector_all, ratio_vector.unsqueeze(0)), dim=0)
    print('成功取出{}个ratio'.format(seed_num))
    return ratio_vector_all

###------------------------------------------extract means of 10000 inverted images------------------------------------------###
def extract_x_mean(seed_num=10000,layer_pos=False):
    # Input : Number of images you want to extract
    # Output : Concatenated means in dim 0
    # base = '/home/shuang/Jupyter_linux/StyleGAN2++/x_mean_vector_10000/'
    base = '/home/shuang/Jupyter_linux/StyleGAN2++/x_mean_all_layer/'
    seed_base = 0
    seed_num = seed_num
    # ratio_vector_all=torch.tensor()
    for i in range(seed_num):
        # print(i)
        seed = seed_base + i
        # file_name = base + str(seed).zfill(5) + '_x_mean_vector.kpl'
        file_name = base + str(seed).zfill(5) + '_x_mean.kpl'
        x_mean_vector = load_variavle(file_name)
        if type(layer_pos) == dict:
            count = 0
            for layer in layer_pos:
                if count:
                    x_mean_tem = torch.cat((x_mean_tem, x_mean_vector[layer_pos[layer][0]:(layer_pos[layer][1])]),
                                           dim=0)
                else:
                    x_mean_tem = x_mean_vector[layer_pos[layer][0]:(layer_pos[layer][1])]
                    count = count + 1
            x_mean_vector = x_mean_tem
        if i == 0:
            x_mean_vector_all = x_mean_vector.unsqueeze(0)
        else:
            x_mean_vector_all = torch.cat((x_mean_vector_all, x_mean_vector.unsqueeze(0)), dim=0)
    print('成功取出{}x_mean'.format(seed_num))
    return x_mean_vector_all

###------------------------------------------extract means of 10000 inverted images------------------------------------------###
def extract_latents_by_index(base_folder,seed_num=10000,layer_pos=False):
    import os
    import numpy as np
    latent_name = base_folder + '/latents_.npy'
    data_npy = np.load(latent_name, allow_pickle=True).item()
    seed_num = seed_num
    idx=0
    for key in data_npy:
        x_mean_vector = torch.tensor(data_npy[key])[0]
    # ratio_vector_all=torch.tensor()

        # x_mean_vector = load_variavle(file_name)
        if type(layer_pos) == dict:
            count = 0
            for layer in layer_pos:
                if count:
                    x_mean_tem = torch.cat((x_mean_tem, x_mean_vector[layer_pos[layer][0]:(layer_pos[layer][1])]),
                                           dim=0)
                else:
                    x_mean_tem = x_mean_vector[layer_pos[layer][0]:(layer_pos[layer][1])]
                    count = count + 1
            x_mean_vector = x_mean_tem
        if idx == 0:
            x_mean_vector_all = x_mean_vector.unsqueeze(0)
            idx=1
        else:
            x_mean_vector_all = torch.cat((x_mean_vector_all, x_mean_vector.unsqueeze(0)), dim=0)
    print('成功取出{}个x_mean'.format(len(data_npy)))
    return x_mean_vector_all
    # for idx in range(len(os.listdir(base_folder))):
    #     if idx > seed_num:
    #         break
    #     file_name = os.path.join(base_folder, str(idx)+'_x_mean_combined.kpl')

###------------------------------------------extract means of 10000 inverted images------------------------------------------###
def extract_x_mean_by_index(base_folder,seed_num=10000,layer_pos=False,zfill=False):
    import os
    seed_base = 0
    seed_num = seed_num
    # ratio_vector_all=torch.tensor()
    for idx in range(len(os.listdir(base_folder))):
        if idx > seed_num:
            break
        file_name = os.path.join(base_folder, str(idx)+'_x_mean_combined.kpl')
        if zfill:
            file_name = os.path.join(base_folder, str(idx).zfill(5) + '_x_mean_combined.kpl')
        x_mean_vector = load_variavle(file_name)
        if type(layer_pos) == dict:
            count = 0
            for layer in layer_pos:
                if count:
                    x_mean_tem = torch.cat((x_mean_tem, x_mean_vector[layer_pos[layer][0]:(layer_pos[layer][1])]),
                                           dim=0)
                else:
                    x_mean_tem = x_mean_vector[layer_pos[layer][0]:(layer_pos[layer][1])]
                    count = count + 1
            x_mean_vector = x_mean_tem
        if idx == 0:
            x_mean_vector_all = x_mean_vector.unsqueeze(0)
        else:
            x_mean_vector_all = torch.cat((x_mean_vector_all, x_mean_vector.unsqueeze(0)), dim=0)
    print('成功取出{}个x_mean'.format(min(len(os.listdir(base_folder)), seed_num)))
    return x_mean_vector_all

###------------------------------------------extract means of 10000 inverted images------------------------------------------###
def extract_x_mean_by_folder_name(base_folder,seed_num=10000,layer_pos=False):
    import os
    # Input : Number of images you want to extract
    # Output : Concatenated means in dim 0
    # base = '/home/shuang/Jupyter_linux/StyleGAN2++/x_mean_vector_10000/'
    seed_base = 0
    seed_num = seed_num
    # ratio_vector_all=torch.tensor()
    for idx, i in enumerate(os.listdir(base_folder)):
        if idx > seed_num:
            break
        file_name = os.path.join(base_folder,i)
        x_mean_vector = load_variavle(file_name)
        if type(layer_pos) == dict:
            count = 0
            for layer in layer_pos:
                if count:
                    x_mean_tem = torch.cat((x_mean_tem, x_mean_vector[layer_pos[layer][0]:(layer_pos[layer][1])]),
                                           dim=0)
                else:
                    x_mean_tem = x_mean_vector[layer_pos[layer][0]:(layer_pos[layer][1])]
                    count = count + 1
            x_mean_vector = x_mean_tem
        if idx == 0:
            x_mean_vector_all = x_mean_vector.unsqueeze(0)
        else:
            x_mean_vector_all = torch.cat((x_mean_vector_all, x_mean_vector.unsqueeze(0)), dim=0)
    print('成功取出{}x_mean'.format(seed_num))
    return x_mean_vector_all
###----------------------------------------------do statistics about how many abnormal features-----------------------------------------------###
def statistic_abnormal(dict_keys,base_file,thres,seed_num=100,zfill=True):
    # dict_keys=['b0_conv1', 'b1_conv0', 'b1_conv1', 'b2_conv0', 'b2_conv1', 'b3_conv0', 'b3_conv1']
    # base='ratio_all_ffhq/'
    seed_base = 0
    # seed_num = 100
    list_abnormal = []
    for i in range(seed_num):
        if (i % 1000) == 0:
            print('SEED:', i)
        seed = seed_base+i
        if zfill:
            file_name = base_file.format(str(seed).zfill(5))
        else:
            file_name = base_file.format(str(seed))
        # file_name=base+str(seed).zfill(5)+'_ratio_all.kpl'
        ratio_all = load_variavle(file_name)
        abnormal_all = 0
        for j in dict_keys:
            layer_abnormal = sum(ratio_all[j].cpu().abs() > thres)
            abnormal_all += layer_abnormal
        list_abnormal.append(abnormal_all.numpy())
    return list_abnormal


print('    Complete!!!')