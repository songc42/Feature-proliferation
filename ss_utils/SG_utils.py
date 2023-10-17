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
    w_dim = 512
    img_channels = 3
    #Instantiate a generator
    exec('import ' + 'training.' + port_model + ' as t')
    t_n = locals()['t']  
    G_copy = t_n.Generator(z_dim, c_dim, w_dim, img_resolution, img_channels).to(device)
    
    #Load pre-trained parameters
    G_copy.load_state_dict(torch.load(dataset + '.pth'))
    return G_copy, port_model, img_resolution, dataset

###----------------------------------------Compute latent z from given seed-------------------------------------------###
def compute_z(seed, device):
    import numpy as np
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, 512)).to(device)
    return z


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
    
###--------------------------------Show mulitiple imgs together---------------------------------###
def show_mul_rows(x_1,num_rows,save_name=False):
    num_cols = int(np.ceil(len(x_1)/num_rows))
    title=['SG2', 'Ours', 'Truncation {} = 0.7'.format(chr(966))]
    for j in range(int(num_cols)):
        for i in range(num_rows):
            img_feature_1 = x_1[int(i*num_cols+j)]
            modi_n = (img_feature_1.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            ax=plt.subplot(num_rows, num_cols, i*num_cols+j+1)
            plt.imshow(modi_n[0].cpu().numpy())
            ax.set_title(title[j])
        plt.axis('off')
    if save_name:
        plt.savefig(save_name)
    plt.show()
    return


# print('    Complete!!!')
