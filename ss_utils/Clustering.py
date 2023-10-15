# from ss_utils.shuang_utils import *
print('Loading Clustering,', end='')
from .shuang_utils import load_variavle
from ss_utils.shuang_utils import load_variavle, save_variable,create_folder

layer_pos = {'b0_conv1': (0, 512), 'b1_conv0': (512, 1024), 'b1_conv1': (1024, 1536),
             'b2_conv0': (1536, 2048), 'b2_conv1': (2048, 2560), 'b3_conv0': (2560, 3072), 'b3_conv1': (3072, 3584),
             'b4_conv0': (3584, 4096), 'b4_conv1': (4096, 4608), 'b5_conv0': (4608, 4864), 'b5_conv1': (4864, 5120),
             'b6_conv0': (5120, 5248),
             'b6_conv1': (5248, 5376), 'b7_conv0': (5376, 5440), 'b7_conv1': (5440, 5504), 'b8_conv0': (5504, 5536),
             'b8_conv1': (5536, 5568)}


###---------------copy imgs based on clustering dictionary, file_name is directory to copy----------------------------###
def copy_imgs_clustering_index(dic_multi, create_file_name, original_file_name, scale=False):
    import os
    import shutil
    from PIL import Image
    # 遍历字典
    for i in dic_multi:
        # 创建新文件夹
        #     create_file='./Affinity_propagation/Preference_500/'+i
        create_file = os.path.join(create_file_name, i)
        os.mkdir(create_file)
        # 遍历每个序号准备移图
        for j in dic_multi[i][0]:
            # ori_IN_name = file_name + 'IN_' + str(j).zfill(5) + '.png'
            # new_name_IN = create_file + '/' + str(j).zfill(5) + '_IN' + '.png'
            #         shutil.copyfile(ori_IN_name,new_name_IN)
            ori_name = os.path.join(original_file_name, str(j) + '.jpg')
            new_name_ori = os.path.join(create_file, str(j) + '.jpg')
            if scale:
                image = Image.open(ori_name)
                new_image = image.resize((128, 128))
                new_image.save(new_name_ori)
            else:
                shutil.copyfile(ori_name, new_name_ori)

###---------------copy imgs based on clustering dictionary, file_name is directory to copy----------------------------###
def copy_inver_ori_imgs_clustering(dic_multi, file_name, scale=False):
    import os
    import shutil
    from PIL import Image
    # 原文件的路径
    # file_name = '/home/shuang/Jupyter_linux/StyleGAN2++/Inversion_images/'
    # 遍历字典
    for i in dic_multi:
        # 创建新文件夹
        #     create_file='./Affinity_propagation/Preference_500/'+i
        create_file = file_name + i +'_all'
        create_folder(create_file)
        # 遍历每个序号准备移图
        for j in dic_multi[i][0]:
            # ori_IN_name = file_name + 'IN_' + str(j).zfill(5) + '.png'
            # new_name_IN = create_file + '/' + str(j).zfill(5) + '_IN' + '.png'
            #         shutil.copyfile(ori_IN_name,new_name_IN)
            ori_in_name = '/home/shuang/Jupyter_linux/StyleGAN2++/Inversion_images/' + 'IN_' + str(
                j).zfill(5)  + '.png'
            new_name_in = create_file + '/' + str(j).zfill(5) + '_IN' + '.png'
            ori_name = '/media/shuang/049EFC289EFC1440/Users/Shuang-Song/Jupyter_file/improved-precision-and-recall-metric-master/FFHQ_10000/' + str(
                j).zfill(5) + '.png'
            new_name_ori = create_file + '/' + str(j).zfill(5) + '.png'
            if scale:
                image = Image.open(ori_name)
                new_image = image.resize((128, 128))
                new_image.save(new_name_ori)
                ###################################
                image = Image.open(ori_name)
                new_image = image.resize((128, 128))
                new_image.save(new_name_ori)
            else:
                shutil.copyfile(ori_name, new_name_ori)
                shutil.copyfile(ori_in_name, new_name_in)
###---------------copy imgs based on clustering dictionary, file_name is directory to copy----------------------------###
def copy_inversion_imgs_clustering(dic_multi, file_name, scale=False):
    import os
    import shutil
    from PIL import Image
    # 原文件的路径
    # file_name = '/home/shuang/Jupyter_linux/StyleGAN2++/Inversion_images/'
    # 遍历字典
    for i in dic_multi:
        # 创建新文件夹
        #     create_file='./Affinity_propagation/Preference_500/'+i
        create_file = file_name + 'IN_' + i
        create_folder(create_file)
        # 遍历每个序号准备移图
        for j in dic_multi[i][0]:
            # ori_IN_name = file_name + 'IN_' + str(j).zfill(5) + '.png'
            # new_name_IN = create_file + '/' + str(j).zfill(5) + '_IN' + '.png'
            #         shutil.copyfile(ori_IN_name,new_name_IN)
            ori_in_name = '/home/shuang/Jupyter_linux/StyleGAN2++/Inversion_images/' + 'IN_' + str(
                j).zfill(5) + '.png'
            new_name_in = create_file + '/' + str(j).zfill(5) + '_IN' + '.png'

            if scale:
                image = Image.open(ori_in_name)
                new_image = image.resize((128, 128))
                new_image.save(new_name_in)
            else:
                shutil.copyfile(ori_in_name, new_name_in)
                # shutil.copyfile(ori_name, new_name_ori)

###---------------copy imgs based on clustering dictionary, file_name is directory to copy----------------------------###
def copy_imgs_clustering(dic_multi, file_name, scale=False):
    import os
    import shutil
    from PIL import Image
    # 原文件的路径
    # file_name = '/home/shuang/Jupyter_linux/StyleGAN2++/Inversion_images/'
    # 遍历字典
    for i in dic_multi:
        # 创建新文件夹
        #     create_file='./Affinity_propagation/Preference_500/'+i
        create_file = file_name + i
        os.mkdir(create_file)
        # 遍历每个序号准备移图
        for j in dic_multi[i][0]:
            # ori_IN_name = file_name + 'IN_' + str(j).zfill(5) + '.png'
            # new_name_IN = create_file + '/' + str(j).zfill(5) + '_IN' + '.png'
            #         shutil.copyfile(ori_IN_name,new_name_IN)
            ori_name = '/media/shuang/049EFC289EFC1440/Users/Shuang-Song/Jupyter_file/improved-precision-and-recall-metric-master/FFHQ_10000/' + str(
                j).zfill(5) + '.png'
            new_name_ori = create_file + '/' + str(j).zfill(5) + '.png'
            if scale:
                image = Image.open(ori_name)
                new_image = image.resize((128, 128))
                new_image.save(new_name_ori)
            else:
                shutil.copyfile(ori_name, new_name_ori)
    # new_name_2=new_name+str(i).zfill(5)+'.png'


###---------------copy imgs based on clustering dictionary, file_name is directory to copy----------------------------###
def copy_DOG_imgs_clustering(dic_multi, file_name, original_file_name, scale=False):
    import os
    import shutil
    from PIL import Image
    # 原文件的路径
    # file_name = '/home/shuang/Jupyter_linux/StyleGAN2++/Inversion_images/'
    lbl_dict = dict(
        n02093754='Australian terrier',
        n02089973='Border terrier',
        n02099601='Samoyed',
        n02087394='Beagle',
        n02105641='Shih-Tzu',
        n02096294='English foxhound',
        n02088364='Rhodesian ridgeback',
        n02115641='Dingo',
        n02111889='Golden retriever',
        n02086240='Old English sheepdog'
    )
    # 遍历字典
    for i in dic_multi:
        # 创建新文件夹
        #     create_file='./Affinity_propagation/Preference_500/'+i
        create_file = file_name + i
        os.mkdir(create_file)
        # 遍历每个序号准备移图
        for j in dic_multi[i][0]:
            # ori_IN_name = file_name + 'IN_' + str(j).zfill(5) + '.png'
            # new_name_IN = create_file + '/' + str(j).zfill(5) + '_IN' + '.png'
            #         shutil.copyfile(ori_IN_name,new_name_IN)
            save_name = original_file_name[j]
            for label in lbl_dict:
                save_name = save_name.replace(str(label), lbl_dict[label])
            ori_name = '/home/shuang/Jupyter_linux/hyperstyle-main/DOG_IMGS/' + original_file_name[j]
            new_name_ori = create_file + '/' + save_name
            if scale:
                image = Image.open(ori_name)
                new_image = image.resize((128, 128))
                new_image.save(new_name_ori)
            else:
                shutil.copyfile(ori_name, new_name_ori)
    # new_name_2=new_name+str(i).zfill(5)+'.png'


###---------------combine and copy imgs of dic_selected to one file_name----------------------------###
def copy_imgs_dic_selected(dic_selected, file_name):
    import os
    import shutil
    # 原文件的路径
    # file_name = '/home/shuang/Jupyter_linux/StyleGAN2++/Inversion_images/'
    # combine their name
    name_combine = ''
    for i in dic_selected:
        name_combine += i + '_'
    create_file = file_name + name_combine
    os.mkdir(create_file)
    # 遍历字典
    for i in dic_selected:
        # 创建新文件夹
        #     create_file='./Affinity_propagation/Preference_500/'+i

        # 遍历每个序号准备移图
        for j in dic_selected[i][0]:
            ori_name = '/media/shuang/049EFC289EFC1440/Users/Shuang-Song/Jupyter_file/improved-precision-and-recall-metric-master/FFHQ_10000/' + str(
                j).zfill(5) + '.png'
            new_name_ori = create_file + '/' + i + '_' + str(j).zfill(5) + '.png'

            shutil.copyfile(ori_name, new_name_ori)


###---------------------copy ori or inverted imgs with abnormal features above threshold to a new file----------------###
def copy_ori_imgs_abnormal(list_abnormal, new_name, thres):
    import shutil
    # 原图dir
    Ori_file = '/media/shuang/049EFC289EFC1440/Users/Shuang-Song/Jupyter_file/improved-precision-and-recall-metric-master/FFHQ_10000/'
    for i in range(len(list_abnormal)):
        if list_abnormal[i] > thres:
            # original img path
            ori_name = Ori_file + str(i).zfill(5) + '.png'
            ori_new_name = new_name + str(i).zfill(5) + '.png'
            shutil.copyfile(ori_name, ori_new_name)


def copy_inverted_imgs_abnormal(list_abnormal, new_name, thres):
    import shutil
    # Inversion img dir
    Inversion_file = '/home/shuang/Jupyter_linux/StyleGAN2++/Inversion_images/'
    for i in range(len(list_abnormal)):
        if list_abnormal[i] > thres:
            # Invertsion img path
            Inversion_name = Inversion_file + 'IN_' + str(i).zfill(5) + '.png'
            Inversion_new_name = new_name + str(i).zfill(5) + '_IN_' + str(list_abnormal[i]) + '.png'
            shutil.copyfile(Inversion_name, Inversion_new_name)


###-------------------------------compute affinity propagation of input vectors---------------------------------------###
def affinity_propagation(ratio_vector_all, distance=-500, appear_count=0):
    # Input : Concatenated vectors
    # Output : dic_multi : Clustering dictionary (class : img index), clustering.labels_ : label of each img, sorted_unique : unique dictionary (classes : corresponding number of existence)
    import numpy as np
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set()
    # from sklearn.datasets.samples_generator import make_blobs
    from sklearn.datasets import make_blobs
    from sklearn.cluster import AffinityPropagation

    # 进行聚类并找出unique的class并计数
    af = AffinityPropagation(preference=distance)
    clustering = af.fit(ratio_vector_all)
    unique, counts = np.unique(clustering.labels_, return_counts=True)
    # 按出现次数从大到小对这些类进行排序
    sorted_lst = sorted(enumerate(counts), key=lambda x: x[1], reverse=True)
    sorted_nums = [x[1] for x in sorted_lst]
    sorted_indices = [x[0] for x in sorted_lst]
    sorted_unique = list(zip(unique[sorted_indices], sorted_nums))
    # 按照类的大小从大到小索引这些的图像序号
    dic_multi = {}
    for idx, i in enumerate(sorted_nums):
        if i > appear_count:
            corres_label = unique[sorted_indices[idx]]
            dic_multi[str(i) + '_' + str(corres_label)] = np.where(clustering.labels_ == corres_label)
    print('Processed {} Affinity, distance is {}.'.format(len(ratio_vector_all), distance))
    compute_clustered_num(sorted_unique)
    return dic_multi, clustering.labels_, sorted_unique


###----------------------------------------compute how many imgs are clustered----------------------------------------###
def compute_clustered_num(sorted_unique):
    num_all = 0
    num_class = 0
    for i in sorted_unique:
        if i[1] > 1:
            num_all += i[1]
            num_class += 1
    print('共有{}个图被聚类了, 聚成了{}类'.format(num_all, num_class))
    return num_all


###-------------------------------Initialize tsne---------------------------------###
from time import time
from sklearn import datasets
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, init='pca', random_state=0)

###-------------------------------Visualize result using dots, same label same color ---------------------------------###
import matplotlib.pyplot as plt


def plot_embedding(data, label, title, class_color, interval=False):
    import numpy as np
    import matplotlib.pyplot as plt
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    # ax = plt.subplot(111)
    num_color = len(class_color)
    for i in range(data.shape[0]):
        if interval:
            if i % interval == 0:
                plt.plot(data[i, 0], data[i, 1], 'o', color=plt.cm.Set1(class_color[label[i]] / (num_color + 1)))

        else:
            plt.plot(data[i, 0], data[i, 1], 'o', color=plt.cm.Set1(class_color[label[i]] / (num_color + 1)))
    #             plt.plot(data[i, 0], data[i, 1], 'o',color=plt.cm.Set1(0.25))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def plot_embedding_value(data, label, title, interval=False):
    import numpy as np
    import matplotlib.pyplot as plt
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    # ax = plt.subplot(111)

    for idx, i in enumerate(label):
        if i == 'Non':
            label[idx] = 0
        else:
            label[idx] = float(label[idx])
    num_color = 1.1 * max(label)
    for i in range(data.shape[0]):
        if interval:
            if i % interval == 0:
                plt.plot(data[i, 0], data[i, 1], 'o', color=plt.cm.Set1(float(label[i]) / num_color))

        else:
            plt.plot(data[i, 0], data[i, 1], 'o', color=plt.cm.Set1(float(label[i]) / num_color))
    #             plt.plot(data[i, 0], data[i, 1], 'o',color=plt.cm.Set1(0.25))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def plot_embedding_text(data, label, title, interval=False, font_size=10, lengend_color=False, s=2, class_color=False):
    import numpy as np
    import matplotlib.pyplot as plt
    import mplcursors
    from matplotlib import image as mpimg
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(18,6))
    ax = plt.subplot(131)
    num_color = len(lengend_color)

    type_x_all = []
    type_y_all = []
    for i in range(num_color):
        type_x_all.append([])
        type_y_all.append([])
    for i in range(data.shape[0]):
        if interval:
            if i % interval == 0:
                type_x_all[int(label[i])].append(data[i, 0])
                type_y_all[int(label[i])].append(data[i, 1])
        else:
            type_x_all[int(label[i])].append(data[i, 0])
            type_y_all[int(label[i])].append(data[i, 1])
    scatter_all = []
    # remove non labels
    non_remove = True

    if non_remove:
        Non_start = 1
        lengend_color_sorted = []
        for idx, i in enumerate(lengend_color):
            if idx != 0:
                lengend_color_sorted.append(lengend_color[i])
    else:
        Non_start = 0
        lengend_color_sorted = lengend_color.values()

    for i in range(Non_start, num_color):
        # scatter_all.append(ax.scatter(type_x_all[i], type_y_all[i], 'o', color=plt.cm.Set1(i+1 / (num_color+1))))
        scatter_all.append(ax.scatter(type_x_all[i], type_y_all[i], marker='o', s=s, color=plt.cm.Set1(i / num_color)))
    ax.legend(scatter_all, lengend_color_sorted, loc=2, bbox_to_anchor=(-0.15, 1.0), borderaxespad=0.,
              fontsize=font_size)
    # for i in range(data.shape[0]):
    #     if interval:
    #         if i % interval == 0:
    #             plt.plot(data[i, 0], data[i, 1], 'o', color=plt.cm.Set1(class_color[label[i]] / (num_color+1)))
    #             plt.text(data[i, 0], data[i, 1],'%.0f' % class_color[label[i]])
    #     else:
    #         plt.plot(data[i, 0], data[i, 1], 'o', color=plt.cm.Set1(class_color[label[i]] / (num_color + 1)))
    #         plt.text(data[i, 0], data[i, 1], '%.0f' % class_color[label[i]])
    #             plt.plot(data[i, 0], data[i, 1], 'o',color=plt.cm.Set1(0.25))

    crs = mplcursors.cursor(hover=False)
    Ori_file = '/media/shuang/049EFC289EFC1440/Users/Shuang-Song/Jupyter_file/improved-precision-and-recall-metric-master/FFHQ_10000/'
    Inversion_file = '/home/shuang/Jupyter_linux/StyleGAN2++/Inversion_images/'

    @crs.connect("add")
    def on_add(sel):
        # index_pic = str(sel.target.index).zfill(5)
        index_pic = str(np.where(data == sel.target)[0][0]).zfill(5)
        # save_variable(index_pic,'where_pos.txt')
        ori_name = Ori_file + index_pic + '.png'
        image1 = mpimg.imread(ori_name)

        Inversion_name = Inversion_file + 'IN_' + index_pic + '.png'
        image2 = mpimg.imread(Inversion_name)

        # plt.figure(1)
        plt.subplot(132)
        plt.imshow(image1)
        plt.title(index_pic+'_original')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(image2)
        plt.title(index_pic + '_inversion')
        plt.axis('off')
        # plt.show()
        # save_variable((sel.target[0], sel.target[1]), '12345.txt')
        # sel.annotation.set_text(labels[sel.target.index])

    # crs.connect("add", lambda sel: sel.annotation.set_text(
    #     'Point {},{}'.format(sel.target[0], sel.target[1])))

    # plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()
    plt.ion()
    return fig


###----------------------------------------------------Filter class---------------------------------------------------###
def filter_class(sorted_unique, filter_range, x_mean_vector_all, affi_labels):
    # 挑出符合filter_range的分类
    main_class = [i[0] for i in sorted_unique if i[1] in filter_range]
    main_index = [i in main_class for i in affi_labels]
    # extract imgs based on main class
    modi_x_mean = x_mean_vector_all[main_index]
    modi_label = affi_labels[main_index]
    class_color = {}
    for idx, i in enumerate(main_class):
        class_color[i] = idx + 1
    return modi_x_mean, modi_label, class_color, main_index


###----------------------------------------------------Filter class---------------------------------------------------###
def filter_dic_color(color_selected, class_color, dic_multi):
    # label_selected
    class_dic = [i for i in class_color]
    label_selected = [class_dic[i - 1] for i in color_selected]
    # dic_selected
    dic_selected = {}
    for i in label_selected:
        for j in dic_multi:
            i_len = len(str(i))
            if j[-i_len:] == str(i) and j[-i_len - 1:-i_len] == '_':
                dic_selected[j] = dic_multi[j]
    return label_selected, dic_selected


###-------------------------------------------Combine dic and give a new name-----------------------------------------###
def combine_dic(dic_selected, combine_name):
    combined_selected = {combine_name: []}
    for i in dic_selected:
        if len(combined_selected[combine_name]):
            combined_selected[combine_name] = np.concatenate((combined_selected[combine_name], dic_selected[i][0]),
                                                             axis=0)
        else:
            combined_selected[combine_name] = dic_selected[i][0]
    return combined_selected


print('    Complete!!!')


###-------------------------------------------Clusering_ACC-----------------------------------------###
def clusering_parameterize(attribute_dic, dic_multi, std_mean=True, parameterize=True):
    import numpy as np
    import copy
    parameterize_dic_all = {'gender': {'Non': 1.5, 'Male': [1, 0], 'Female': [0, 1]},
                            'glasses': {'Non': 0, 'NoGlasses': [1, 0, 0, 0], 'ReadingGlasses': [0, 1, 0, 0],
                                        'Sunglasses': [0, 0, 1, 0],
                                        'SwimmingGoggles': [0, 0, 0, 1]},
                            'age': {'Non': 0, 'Baby': 1 / 6, 'Child': 2 / 6, 'Teenage': 3 / 6, 'Adult': 4 / 6,
                                    'Seniors': 5 / 6, 'Elderly': 6 / 6}
                            }
    each_attribute_cluster_dic = {}
    dic_multi_modi = {}
    for attribute in attribute_dic:
        if attribute in parameterize_dic_all:
            parameterize_dic = parameterize_dic_all[attribute]
        if not parameterize:
            parameterize_dic_all = {}
        # 取出该class的所有结果
        Attributes = load_variavle('Attributes_list/{}.kpl'.format(attribute))
        dic_multi_attri = {}
        for i in dic_multi:
            dic_multi_modi[i] = []
            #         if attribute
            #         tem_attribue_val=np.empty([len(dic_multi[i][0]),])
            tem_attribue_val = []
            # 取出分类的序号
            for idx, j in enumerate(dic_multi[i][0]):
                if Attributes[j] == 'Non':
                    continue
                #             Attributes[j]
                #             tem_attribue_val[idx] = parameterize_dic[Attributes[j]]
                dic_multi_modi[i].append(j)
                if attribute in parameterize_dic_all:
                    tem_attribue_val.append(parameterize_dic[Attributes[j]])
                else:
                    tem_attribue_val.append(Attributes[j])
            if std_mean:
                dic_multi_attri[i] = (
                np.mean(np.std(np.array((tem_attribue_val)), axis=0)), np.mean(np.array(tem_attribue_val), axis=0))
            else:
                dic_multi_attri[i] = tem_attribue_val
        each_attribute_cluster_dic[attribute] = dic_multi_attri  # 该分类下所有分好类的均值和方差
        #
    each_cluster_attribute_dic = {}
    each_cluster_attribute_dex_dic = copy.deepcopy(each_cluster_attribute_dic)
    for i in dic_multi:
        each_cluster_attribute_dic[i] = {}
        each_cluster_attribute_dex_dic[i] = {}
        for attribute in attribute_dic:
            each_cluster_attribute_dic[i][attribute] = each_attribute_cluster_dic[attribute][i]

            combine_attri_dex = []
            for k in range(len(dic_multi_modi[i])):
                combine_attri_dex.append((each_attribute_cluster_dic[attribute][i][k], dic_multi_modi[i][k]))
            each_cluster_attribute_dex_dic[i][attribute] = combine_attri_dex
    return each_attribute_cluster_dic, each_cluster_attribute_dic, each_cluster_attribute_dex_dic


def clusering_ACC(each_cluster_attribute_dic, class_eva_dic, ignore_num_thres=3):
    overall_ACC = {}
    individual_acc = {}
    for class_eva in class_eva_dic:
        num_total = 0
        num_acc_total = 0
        individual_acc[class_eva] = {}
        # 确定有三个及以上的类
        for i in each_cluster_attribute_dic:
            cluster_attribute = each_cluster_attribute_dic[i][class_eva]
            num_cluster = len(cluster_attribute)
            if num_cluster < ignore_num_thres:
                continue
            # 确定这个类的主属性
            main_attribute = max(cluster_attribute, key=cluster_attribute.count)
            num_acc = cluster_attribute.count(main_attribute)
            individual_acc[class_eva][i] = num_acc / num_cluster
            # 加上数目
            num_total = num_total + num_cluster
            num_acc_total = num_acc_total + num_acc
        overall_ACC[class_eva] = num_acc_total / num_total
    return overall_ACC, individual_acc


###-------------------------------------Extract label from annotated FFHQ dataset-------------------------------------###
def attribute_label(attribute, main_index, mark_size=2, show_img=False, font_size=10, result=False, interval=2):
    import numpy as np
    Attributes = load_variavle('Attributes_list/{}.kpl'.format(attribute))
    FFHQ_label = np.empty([10000, ])

    transfer_dic_all = {'gender': {'Non': 0, 'Male': 1, 'Female': 2},
                        'glasses': {'Non': 0, 'No Glasses': 1, 'Reading Glasses': 2, 'Sun Glasses': 3,
                                    'Swimming Goggles': 4},
                        '2_glasses': {'Non': 0, 'No Glasses': 1, 'Glasses': 2},
                        'age': {'Non': 0, 'Baby': 1, 'Child': 2, 'Teenage': 3, 'Adult': 4, 'Seniors': 5, 'Elderly': 6},
                        'DOG': {'ILSVRC2012': 0, 'n02088364': 1, 'n02096294': 2, 'n02093754': 3, 'n02089973': 4,
                                'n02105641': 5, 'n02115641': 6, 'n02111889': 7, 'n02086240': 8,
                                'n02087394': 9, 'n02099601': 10},
                        'LFW': {'Serena_Williams': 0, 'Junichiro_Koizumi': 1, 'George_W': 2, 'Jean_Chretien': 3,
                                'Hugo_Chavez': 4, 'John_Ashcroft': 5, 'Tony_Blair': 6, 'Donald_Rumsfeld': 7,
                                'Colin_Powell': 8, 'Gerhard_Schroeder': 9, 'Jacques_Chirac': 10, 'Vladimir_Putin': 11,
                                'Ariel_Sharon': 12},
                        'IP102': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
                        }
    class_color = {}
    lengend_color = {}
    if attribute.split('_')[-1] in transfer_dic_all:
        # if attribute.split('_')[-1] in transfer_dic_all:
        transfer_dic = transfer_dic_all[attribute]
        for i in transfer_dic:
            class_color[transfer_dic[i]] = transfer_dic[i] + 1
            lengend_color[transfer_dic[i]] = i
        # for i in range(len(transfer_dic)):
        # class_color[i] = i + 1
        print()
        for i in range(len(result)):
            FFHQ_label[i] = transfer_dic[Attributes[i]]
        if main_index == False:
            modi_FFHQ_label = FFHQ_label
        else:
            modi_FFHQ_label = FFHQ_label[main_index]
    else:

        modi_FFHQ_label = Attributes
    title = 't-SNE embedding of Attributes: {}'.format(attribute)

    fig = 0

    if show_img:
        if attribute.split('_')[-1] in transfer_dic_all:
            fig = plot_embedding_text(result, modi_FFHQ_label, title, s=mark_size, font_size=font_size,
                                      interval=interval, lengend_color=lengend_color)
        else:
            fig = plot_embedding_value(result, modi_FFHQ_label, title, interval=False)
    return modi_FFHQ_label, lengend_color, title, fig


###-------------------------------------Extract label from annotated FFHQ dataset-------------------------------------###
def visulaization_selection(label_visual, modi_FFHQ_label, modi_label, FFHQ_class_color, class_color, class_label):
    if label_visual:
        visual_label = modi_FFHQ_label
        visual_color = FFHQ_class_color
        title = 't-SNE embedding of Attributes: {}'.format(class_label)
    else:
        visual_label = modi_label
        visual_color = class_color
        title = 't-SNE embedding of all '
    return visual_color, visual_label, title


###-------------------------------------Extract label from annotated FFHQ dataset-------------------------------------###
def Attribute_label(attribute):
    import numpy as np
    Attributes = load_variavle('Attributes_list/{}.kpl'.format(attribute))
    FFHQ_label = np.empty([70000, ])
    transfer_dic_all = {'gender': {'Non': 0, 'male': 1, 'female': 2},
                        'glasses': {'Non': 0, 'NoGlasses': 1, 'ReadingGlasses': 2, 'Sunglasses': 3,
                                    'SwimmingGoggles': 4}
                        }
    if attribute in transfer_dic_all:
        transfer_dic = transfer_dic_all[attribute]
    class_color = {}
    lengend_color = {}
    for i in transfer_dic:
        class_color[transfer_dic[i]] = transfer_dic[i] + 1
        lengend_color[transfer_dic[i]] = i
    # for i in range(len(transfer_dic)):
    # class_color[i] = i + 1

    for i in range(70000):
        if attribute in transfer_dic_all:
            FFHQ_label[i] = transfer_dic[Attributes[i]]
        else:
            FFHQ_label[i] = Attributes[i]

    return FFHQ_label, class_color, lengend_color


###---------------------------------Normalise list with 'Non' to (0,1) mean and std-----------------------------------###
def nomarlise_non(file_name):
    import numpy as np
    Attributes_list = load_variavle(file_name)
    Attributes_list_non = []
    for i in Attributes_list:
        if i == 'Non':
            continue
        Attributes_list_non.append(i)
    it_mean = np.mean(Attributes_list_non)
    it_std = np.std(Attributes_list_non)
    for i in range(len(Attributes_list)):
        if Attributes_list[i] == 'Non':
            continue
        Attributes_list[i] = (Attributes_list[i] - it_mean) / it_std
    return Attributes_list
