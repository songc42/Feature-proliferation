print('Loading SG_utils,', end='')
import torch
from ss_utils.shuang_utils import save_variable, load_variavle
import matplotlib.pyplot as plt
import numpy as np
###-------------------------------------------------------------------------------------------Functions to run a SG model-----------------------------------------------------------------------------------------###


###---------------------------------------Select a model you want to launch-------------------------------------------###
def model_select(a, key):
    device = torch.device('cuda')
    try:
        port_model, img_resolution, dataset = a[key]
        # aa=a[key]
    except:
        raise Exception("Sorry, Wrong key")
        return 0
    z_dim = 512
    c_dim = 0
    if dataset == 'CIFAR10':
        print('CIFAR10')
        c_dim = 0
    w_dim = 512
    img_channels = 3
    exec('import ' + 'training.' + port_model + ' as t')
    t_n = locals()['t']  ############################point
    if dataset=='HUMAN':
        G_copy = t_n.Generator(z_dim, c_dim, w_dim, 0,img_resolution, img_channels).to(device)
    else:
        G_copy = t_n.Generator(z_dim, c_dim, w_dim, img_resolution, img_channels).to(device)
    G_copy.load_state_dict(torch.load(dataset + '.pth'))
    return G_copy, port_model, img_resolution, dataset

###----------------------------------------Compute latent z from given seed-------------------------------------------###
def compute_z(seed, device):
    import numpy as np
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, 512)).to(device)
    return z

###---------------------------------Extract mean and value from a certain key(layer)----------------------------------###
def extract_m_vari(dataset, key):
    m_all = load_variavle(dataset + '_m_var/conv' + key[-1] + '_m_b' + key[1] + '.txt')
    vari_all = load_variavle(dataset + '_m_var/conv' + key[-1] + '_vari_b' + key[1] + '.txt')
    # print(dataset+'_m_var/conv1_vari_b' + key + '.txt')
    return m_all, vari_all

###------------------------------Convert generated imgs of numpy to those of PIL.img----------------------------------###
def convert_to_img(img, imshow=False, save_file=None):
    import PIL.Image
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].detach().cpu().numpy(), 'RGB')
    if imshow:
        plt.axis('off')
        plt.imshow(img)
        plt.show()
    if save_file:
        img.save(save_file)
    return img


###---------------------------------------------------------------------------------------Functions to postprocess SG model---------------------------------------------------------------------------------------###

###--------------------------------Compute mean and standard deviation of each layer----------------------------------###
def layer_mean_std(layer_dic,file_name,save_file):
	# Input : layer_dic : a list of name of each layer, file_name : a string of name of loaded data(missing index), save_file : filename to save.
	# Output :
    for key in layer_dic:
        print('Processing key:',key)
        #     key='b7_conv1'
        for i in range(1000):
            x_mean = load_variavle(file_name.format(i))[key]
            if i == 0:
                x_mean_all = x_mean.unsqueeze(0)
            else:
                x_mean_all = torch.cat((x_mean_all, x_mean.unsqueeze(0)))
        std, mean = torch.std_mean(x_mean_all, dim=0, keepdim=True)
        load_variavle(std, save_file + key + '_std.kpl')
        load_variavle(mean, save_file + key + '_mean.kpl')
    print("!complete mean and std save!")
    return 0

###--------------------------------Show mulitiple imgs together---------------------------------###
def show_mul(x_1):
    num = len(x_1)
    for i in range(num):
        img_feature_1 = x_1[i]
        modi_n = (img_feature_1.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        plt.subplot(1, num, i + 1)
        plt.imshow(modi_n[0].cpu().numpy())
        plt.axis('off')
    plt.show()
def show_heat_mul(x_1):
    num = len(x_1)
    for i in range(num):
        img_feature_1 = x_1[i].detach().cpu().numpy()
        plt.subplot(1, num, i + 1)
        ax = sns.heatmap(img_feature_1, center=0, square=True, cbar_kws={"orientation": "vertical"})
        plt.axis('off')
    plt.show()
def show_heat_2(x_1, x_2):
    img_feature_1 = x_1.detach().cpu().numpy()
    img_feature_2 = x_2.detach().cpu().numpy()
    plt.subplot(1, 2, 1)
    #   ax = sns.heatmap(img_feature_1[0], center=0)
    ax = sns.heatmap(img_feature_1, center=0, square=True)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    #   ax = sns.heatmap(img_feature_2[0], center=0)
    ax = sns.heatmap(img_feature_2, center=0, square=True)
    plt.axis('off')
    plt.show()



# print('    Complete!!!')