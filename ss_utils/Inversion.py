print('Loading Inversion utilis,', end='')
import PIL.Image

from ss_utils.shuang_utils import load_variavle, save_variable
from ss_utils.SG_utils import *
from ss_utils.SG2 import model_dic, device
import torch

###---------------------------Perform SG2 inversion form given ws(it has to be in form of 3-d)------------------------###
def inversion_ws(ws, save_name=False, G_copy=None):
    Key_all = {'modify': False, 'modify_0': False, 'modify_1': False, 'record_variation': False, 'heat_display': False,
               'modify_num': False, 'modi_index_all': False, 'ratio_all': False, 'x_modi_yet': False, 'x_all': False, 'x_mean':True}
    save_variable(Key_all, 'Key_all.kpl')
    if G_copy == None:
        Chosed_model = 'ethics_inversion'
        G_copy, import_model, img_resolution, dataset = model_select(model_dic, Chosed_model)
    label = torch.zeros([1, G_copy.c_dim], device=device)
    truncation_psi = 1
    noise_mode = 'const'
    heat_display = 0
    img, dic_para = G_copy(ws, label, dataset, heat_display, truncation_psi=truncation_psi, noise_mode=noise_mode)
    # img_y = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # img_y = img_y.detach().cpu().numpy()
    # img_y = PIL.Image.fromarray(img_y[0], 'RGB')
    # img_y.save(save_name)#'Inversion_images/IN_' + str(seed) + '.png'
    #     save_variable(dic_para['ratio_all'],'ratio_all_ffhq/'+str(seed)+'_ratio_all.kpl')
    #     ratio_all_dic[str(seed)]=dic_para['ratio_all']
    # save_variable(dic_para['ratio_all'],'ratio_all_dic/'+str(seed)+'_ratio_all_dic.kpl')
    return img, dic_para





print('    Complete!!!')
