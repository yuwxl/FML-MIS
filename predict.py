import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from networks.unet2d import Unet2D
from utils.util import _eval_dice, _eval_haus, _connectivity_region_analysis, parse_fn_haus

def get_coutour_sample(y_true):
    # print("y_true shape",y_true.shape)
    disc_mask = np.expand_dims(y_true[..., 0], axis=2)

    disc_erosion = ndimage.binary_erosion(disc_mask[..., 0], iterations=1).astype(disc_mask.dtype)
    disc_dilation = ndimage.binary_dilation(disc_mask[..., 0], iterations=5).astype(disc_mask.dtype)
    disc_contour = np.expand_dims(disc_mask[..., 0] - disc_erosion, axis = 2)
    disc_bg = np.expand_dims(disc_dilation - disc_mask[..., 0], axis = 2)
    cup_mask = np.expand_dims(y_true[..., 1], axis=2)

    cup_erosion = ndimage.binary_erosion(cup_mask[..., 0], iterations=1).astype(cup_mask.dtype)
    cup_dilation = ndimage.binary_dilation(cup_mask[..., 0], iterations=5).astype(cup_mask.dtype)
    cup_contour = np.expand_dims(cup_mask[..., 0] - cup_erosion, axis = 2)
    cup_bg = np.expand_dims(cup_dilation - cup_mask[..., 0], axis = 2)

    return [disc_contour, disc_bg, cup_contour, cup_bg]


def draw_image(image):
    # print("in draw",image)
    plt.imshow(image, cmap='gray')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

    plt.xticks([])
    plt.yticks([])

    return 0

def _save_image(img, gth, pred, out_folder, out_name):

    np.save(out_folder+'/'+out_name+'_img.npy',img)
    np.save(out_folder+'/'+out_name+'_pred.npy',pred)
    np.save(out_folder+'/'+out_name+'_gth.npy',gth)

    return 0

def test(net,img):

    # net = net.cuda
    net.train()

    dice_array = []
    haus_array = []

    data = img
    image = data[..., :3]  # np.expand_dims(data[..., :3].transpose(2, 0, 1), axis=0)
    mask = data[..., 3:]  # np.expand_dims(data[..., 3:].transpose(2, 0, 1), axis=0)
    mask = np.expand_dims(mask.transpose(2, 0, 1), axis=0)
    pred_y_list = []

    print("网络输入", image.shape)
    image_test = np.expand_dims(image.transpose(2, 0, 1), axis=0)
    image_test = torch.from_numpy(image_test).float()
    print("网络输入2",image_test.shape)
    logit, pred, _ = net(image_test)

    print("网络输出",pred.shape)
    # print(pred)
    # print(pred.shape)
    pred_y = pred.cpu().detach().numpy()


    pred_y[pred_y > 0.75] = 1
    pred_y[pred_y < 0.75] = 0

    # print("pred_y",pred_y)
    # print(pred_y.shape)

    pred_y_0 = pred_y[:, 0:1, ...]
    pred_y_1 = pred_y[:, 1:, ...]
    processed_pred_y_0 = _connectivity_region_analysis(pred_y_0)
    processed_pred_y_1 = _connectivity_region_analysis(pred_y_1)
    processed_pred_y = np.concatenate([processed_pred_y_0, processed_pred_y_1], axis=1)
    dice_subject = _eval_dice(mask, processed_pred_y)
    haus_subject = _eval_haus(mask, processed_pred_y)
    dice_array.append(dice_subject)
    haus_array.append(haus_subject)
    dice_array = np.array(dice_array)
    haus_array = np.array(haus_array)

    dice_avg = np.mean(dice_array, axis=0).tolist()
    haus_avg = np.mean(haus_array, axis=0).tolist()

    return dice_avg, haus_avg,image.transpose(2, 0, 1),mask[0],pred_y[0]

if __name__ == '__main__':

#--------------------------------------------------
    # 1. 加载待测图片 img,带标签,不带标签
    # 2. 加载模型 model
    # 3. 预测，生成三张图 :原图，标签图，预测图
    # 4. 展示
#---------------------------------------------------

# 1. 加载待测图片和标签,制作384 * 384 * 5 规范输入numpy形式
    img_path = "./finalBox/raw/g0001.png"
    mask_path = "./finalBox/mask/g0001.png"


    img_trg = Image.open(img_path)      #得到 原始图像的 numpy形式
    img_trg = img_trg.resize((384, 384), Image.BICUBIC)
    img_np = np.asarray(img_trg)

    mask_trg = Image.open(mask_path)        #得到 原始图像的掩码的numpy形式
    mask_trg = mask_trg.resize((384, 384), Image.BICUBIC)
    mask_np = np.asarray(mask_trg)

    # 从掩码 切片
    disk_mask = mask_np[..., 0:1].copy()
    cup_mask = mask_np[..., 0:1].copy()

    # 提取盘 掩码
    disk_mask[disk_mask == 255] = 0
    disk_mask[disk_mask == 0] = 0
    disk_mask[disk_mask != 0] = 1

    # 提取杯掩码
    cup_mask = np.where(cup_mask == 0, 1, 0)

    # 合并，最终 npy 为 384 * 384 * 5
    trg_img_np = np.concatenate((img_np, disk_mask, cup_mask), axis=2)


# 2. 加载模型
    model = Unet2D()
    model_path = "trained_model/xxx.pth"
    # test_net = test_net.cuda()
    device = torch.device("cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))


# 3. 预测，生成三张图 :原图，标签图，预测图,以及分割评估参数
    dice, haus,img,gth,pred = test(model,trg_img_np)
    print(dice[0],dice[1])
    print(haus[0],haus[1])
    # print(type(img))
# 4. 展示
#------------------------------------------------------
    plt.figure(figsize=(2,3))
    # img = np.load('../../output/epiGpu_result/prediction/3_sample1.npy_img.npy')
    imgraw = img.transpose((1, 2, 0)).copy()

    plt.subplot(1, 3, 1)
    draw_image(imgraw)
    plt.xlabel("rawIMG", fontsize=10)

# 真实
#     img_np = np.load('../../output/epiGpu_result/prediction/3_sample1.npy_gth.npy')
    mask_patch = gth.transpose((1,2,0))
    disc_contour, disc_bg, cup_contour, cup_bg = get_coutour_sample(mask_patch)
    # image = np.concatenate((disc_contour, disc_contour, disc_contour), axis=2)
    image = np.concatenate((disc_bg, disc_bg, disc_bg), axis=2)
    imgraw[:, :, :][image[:, :, :] > 0] = 255
    plt.subplot(1, 3, 2)
    draw_image(imgraw)
    plt.xlabel("gthIMG", fontsize=10)
# 预测
#     np.set_printoptions(threshold=np.inf)
#     img_np2 = np.load('../../output/epiGpu_result/prediction/3_sample1.npy_pred.npy')
    mask_patch = pred.transpose((1,2,0))
    disc_contour, disc_bg, cup_contour, cup_bg = get_coutour_sample(mask_patch)
    image = np.concatenate((disc_bg, disc_bg, disc_bg), axis=2)
    imgraw = img.transpose((1, 2, 0)).copy()
    imgraw[:, :, :][image[:, :, :] > 0] = 255
    plt.subplot(1, 3, 3)
    draw_image(imgraw)
    plt.xlabel("preIMG", fontsize=10)
    plt.show()
