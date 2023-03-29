import glob
import os
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def extract_amp_spectrum(trg_img):
    fft_trg_np = np.fft.fft2(trg_img, axes=(-2, -1))
    amp_target, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    return amp_target


def draw_image(image):
    # print("in draw",image)
    plt.imshow(image, cmap='gray')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

    plt.xticks([])
    plt.yticks([])

    return 0

def amp_spectrum_swap(amp_local, amp_target, L=0.1):
    a_local = np.fft.fftshift(amp_local, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_target, axes=(-2, -1))

    _, h, w = a_local.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1
    ratio = random.randint(1,10)/10
    # ratio = 0
    a_local[:, h1:h2, w1:w2] = a_local[:, h1:h2, w1:w2] * ratio + a_trg[:, h1:h2, w1:w2] * (1 - ratio)
    a_local = np.fft.ifftshift(a_local, axes=(-2, -1))
    return a_local

def freq_space_interpolation(local_img, amp_target, L=0, ratio=0):
    local_img_np = local_img
    # print("local_img.shape",local_img.shape)
    # get fft of local sample
    fft_local_np = np.fft.fft2(local_img_np, axes=(-2, -1))

    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    # swap the amplitude part of local image with target amplitude spectrum
    # print("inter1",amp_local.shape)
    # print("inter2",amp_target.shape)
    amp_local_ = amp_spectrum_swap(amp_local, amp_target, L=L)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp(1j * pha_local)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(-2, -1))
    local_in_trg = np.real(local_in_trg)

    return local_in_trg


client_num = 4
client_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
client_data_list = []
len_total = 0
len_list = []
# freq_site_index = [0,1,3]
freq_site_index = [0,1,2,3]
# 准备测试的数据，可不用，直接在测试代码里面跑
for client_idx in range(client_num):
    client_data_list.append(glob.glob('./Datafortrans/{}/*'.format(client_name[client_idx])))
    len_total += len(client_data_list[client_idx])
    len_list.append(len_total)

L = 0.003
# print(len(client_data_list[0]))
plt.figure(figsize=(10,5))
for i in freq_site_index:
    k = 0
    for image_path in client_data_list[2]:
        tar_freq = client_data_list[i]
        # print(len(tar_freq))
        index = random.randint(0,len(tar_freq)-1)
        #本地图片
        im_local = Image.open(image_path)

        im_local = im_local.resize((384, 384), Image.BICUBIC)
        im_local = np.asarray(im_local, np.float32)
        im_local = im_local.transpose((2, 0, 1))  # 把图片变成 3 * 384 * 384 为啥？？
        # print(im_local.shape)
        # plt.subplot(1,3,1)
        # draw_image((im_local / 255).transpose((1, 2, 0)))
        # #draw_image((im_local / 255))
        # plt.xlabel("Local Image", fontsize=7)

        #其他客户端图片
        tar_Img_path = tar_freq[index]
        im_trg = Image.open(tar_Img_path)
        im_trg = im_trg.resize( (384,384), Image.BICUBIC )
        im_trg = np.asarray(im_trg, np.float32)
        im_trg = im_trg.transpose((2, 0, 1))
        # plt.subplot(1, 3, 2)
        # draw_image((im_trg / 255).transpose((1, 2, 0)))
        # plt.xlabel("Target Image", fontsize=7)


        amp_target = extract_amp_spectrum(im_trg)   #提取幅度谱
        local_in_trg = freq_space_interpolation(im_local, amp_target, L=L)
        local_in_trg = local_in_trg.transpose((1, 2, 0))
        local_in_trg = (np.clip(local_in_trg / 255, 0, 1))
        # print(type(local_in_trg))
        # print(local_in_trg.shape)
        # im = Image.fromarray(local_in_trg.astype(np.uint8))
        # print(im.size)
        # path = "./trans/Domain{}/".format(i+1)
        path = "./random_trans/Domain{}/".format(i + 1)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.imsave(path+"{}.jpg".format(k),local_in_trg,cmap='gray')
        k = k+1


        # plt.subplot(1, 3, 3)
        # # plt.imshow(local_in_trg,cmap='gray')
        # draw_image(local_in_trg)
        # plt.xlabel("Trans Image", fontsize=7)
        # plt.show()

    # tar_freq = np.load(tar_freq)
        # L1 = random.randint(2,5)/1000.0
    # image_patch_freq_1 = source_to_target_freq(image_patch, tar_freq[...], L=0)