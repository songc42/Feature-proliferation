print('Loading CIFAR_10,', end='')
import sys
sys.path.append('/home/shuang/Documents/CVPR_docs/CVPR_dataset/')
from .shuang_utils import load_variavle
from ss_utils.shuang_utils import load_variavle, save_variable,create_folder
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data_batch(num):
    data_folder = 'cifar-10-python/cifar-10-batches-py/data_batch_'+str(num)
    dic_all = unpickle(data_folder)
    return dic_all

def load_imgs(num, show=False):

    import matplotlib.pyplot as plt
    from matplotlib import image as mpimg
    save_folder = 'Extracted_CIFAR_10/batch_all/'+str(num)+'.jpg'
    image2 = mpimg.imread(save_folder)
    if show:
        plt.imshow(image2)
    return image2

def load_batches_meta():
    data = unpickle('cifar-10-python/cifar-10-batches-py/batches.meta')[b'label_names']
    return data

def load_labels(num,out_all=False,meta_class=True):
    label_all = load_variavle('Extracted_CIFAR_10/labels/batch_all.kpl')
    label_num = label_all[num]
    if meta_class:
        label_num = {label_num: load_batches_meta()[label_num]}
    if out_all:
        label_num = label_all
    return label_num