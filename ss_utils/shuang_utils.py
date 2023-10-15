
print('Loading shuang_utils,',end='')
#import torch
# import numpy as np
###---------------------------------------------Load and save file----------------------------------------------------###
def load_variavle(filename):
    import pickle
    f=open(filename,'rb')
    r=pickle.load(f)
    #r=torch.load(f，map_location=torch.device('cpu'))
    f.close()
    return r
def save_variable(v,filename):
    import pickle
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

###---------------------------------print all elements with index prefixed to itself----------------------------------###
def print_all(v,index_display=False):
    if index_display:
        for idx,i in enumerate(v):
            print('{}:{}'.format(idx,i))
    else:
        for i in v:
            print(i)

###----------------------------------------------------创建文件夹----------------------------------------------###
def create_folder(folder_path):
    import os
    folder = folder_path
    if not os.path.exists(folder_path):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder_path)
###----------------------------------------------多个文件夹下文件合并-----------------------------------------------------###
def combine_folders(filePath,savePath):
    import os
    import shutil
    # filePath='/home/shuang/Downloads/AAAI_DATASET/lfw-deepfunneled_13'
    # savePath = '/home/shuang/Downloads/AAAI_DATASET/deepfunneled_13_extracted'
    for i in os.listdir(filePath):
        dir_path = os.path.join(filePath,i)
        for j in os.listdir(dir_path):
            ori_name = os.path.join(dir_path,j)
            new_name_ori=os.path.join(savePath,j)
            shutil.copyfile(ori_name, new_name_ori)

###------------------------------Load pretrained model and save to .pth file for late use-----------------------------###
def Load_save_parameter(input_name,save_name):
    import torch
    import pickle
    with open(input_name, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
        torch.save(G.state_dict(), save_name)
    return 0


###-----------------------------------------------------utils---------------------------------------------------------###
import math
def list_sqrt(a):
  c = []
  for i in range(len(a)):
      c.append(math.sqrt(a[i]))
  return c

def list_add(a,b):
  for i in range(len(a)):
    a[i] += b[i]
  return a

def list_sub(a,b):
  c = []
  for i in range(len(a)):
      c.append(a[i]-b[i])
  return c

def list_sub_square(a,b):
  c = []
  for i in range(len(a)):
      c.append((a[i]-b[i])**2)
  return c

def list_div_list(a,b):
  c = []
  for i in range(len(a)):
      c.append(a[i]/b[i])
  return c

def list_div_num(a,b):
  c = []
  for i in range(len(a)):
      c.append(a[i]/b)
  return c







print('    Complete!!!')
