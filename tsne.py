import glob
import os
import numpy as np
import cv2
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.manifold import TSNE
def get_data():
    client_num = 4
    client_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
    client_data_list = []
    len_total = 0
    len_list = []

    for client_idx in range(client_num):
        # client_data_list.append(glob.glob('./Datafortrans/{}/*'.format(client_name[client_idx])))
        # client_data_list.append(glob.glob('./trans/{}/*'.format(client_name[client_idx])))
        client_data_list.append(glob.glob('./random_trans/{}/*'.format(client_name[client_idx])))
        len_total+=len(client_data_list[client_idx])
        len_list.append(len_total)

    print(len_list)
    data=np.zeros((len_total,40000)) #初始化一个np.array数组用于存数据
    label=np.zeros((len_total,)) #初始化一个np.array数组用于存数据

    for k in range(0,len_list[0]):
        label[k] = 1
    for k in range(len_list[0],len_list[1]):
        label[k] = 2
    for k in range(len_list[1],len_list[2]):
        label[k] = 3
    for k in range(len_list[2],len_list[3]):
        label[k] = 4

    # print(label)

    index = 0
    for i in range(4):
         for image_path in client_data_list[i]:
            # image_path=os.path.join(Input_path,Image_names[i])
            # print(image_path)
            img=cv2.imread(image_path)
            img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img=cv2.resize(img_gray,(200,200))
            img=img.reshape(1,40000)
            data[index]=img
            index+=1
            n_samples, n_features = data.shape

            # print(data.shape)
            # print(len(data))
    print(index)
    return data, label, n_samples, n_features

def plot_embedding_2D(data, label, title):
    # print(len(data))
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    # fig = plt.figure(figsize=(8, 6))  # 新建画布
    # print(data.shape[0])
    for i in range(data.shape[0]):
        # print(label[i])
        c = label[i].astype(int)
        # print(c)
        # print(type(c))
        color = plt.cm.Set1(c)
        # print('color', i, color)
        # print(data[i, 0], data[i, 1])
        # plt.text(data[i, 0], data[i, 1], str(label[i]),
        #          color=color,
        #          fontdict={'weight': 'bold', 'size': 9})
        plt.scatter(data[i, 0], data[i, 1], c=color, marker='o')
    plt.xticks([])
    plt.yticks([])
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    # plt.xlabel("无插值")
    # plt.xlabel("λ ∈ [0.0,1.0]")
    # plt.xlabel("w/o CFSI")
    # plt.text(0.0, 48,"t-SNE")
    # plt.title(title)

    return fig

def plot_embedding_3D(data,label,title):
    x_min, x_max = np.min(data,axis=0), np.max(data,axis=0)
    data = (data- x_min) / (x_max - x_min)
    ax = plt.figure().add_subplot(111,projection='3d')
    for i in range(data.shape[0]):
        c = label[i].astype(int)
        color = plt.cm.Set1(c)
        ax.scatter(data[i, 0], data[i, 1], data[i,2], marker="o",c=color)
        plt.xlabel("λ ∈ [0.0,1.0]")

        # ax.text(data[i, 0], data[i, 1], data[i,2],"o", color=color,fontdict={'weight': 'bold', 'size': 9})
    return ax

def get_data2():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features

#主函数
def main():
    data, label, n_samples, n_features = get_data()
    print('Begining......')
    tsne_2D = TSNE(n_components=2, init='pca', random_state=501) #调用TSNE
    result_2D = tsne_2D.fit_transform(data)
    # tsne_3D = TSNE(n_components=3, init='pca', random_state=0)
    # result_3D = tsne_3D.fit_transform(data)
    print('Finished......')
    fig1 = plot_embedding_2D(result_2D, label,'t-SNE')
    # # print(fig1)
    # # plt.show(fig1)
    fig1.show()
    plt.show()
    # fig2 = plot_embedding_3D(result_3D, label,'t-SNE')
    # fig2.show()
    # plt.show()
if __name__ == '__main__':

    main()
