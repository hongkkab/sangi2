# -*-coding:UTF-8-*-
import os
import scipy.io
import numpy as np
import glob
import torch.utils.data as data
import scipy.misc
from PIL import Image
import cv2
import Mytransforms
import sys
import torchvision.transforms as transforms
#import Mytransforms

# crop data (368*368)*2700

def read_sangi_data_file(root_dir):
    image_arr = np.array(glob.glob(os.path.join(root_dir, 'Cooler_crop/*.jpg')))
    image_nums_arr = np.array([float(s.rsplit('.')[-2][-5:]) for s in image_arr])
    sorted_image_arr = image_arr[np.argsort(image_nums_arr)]
    return sorted_image_arr.tolist()

def read_sangi_mat_file(root_dir, img_list):
    
    mat_arr = scipy.io.loadmat(os.path.join(root_dir, 'Cooler_crop_joint.mat'))['Cooler_crop_joint']
    # (14,3, 2700)
#    lms = mat_arr.transpose([2, 1, 0])
    kpts = mat_arr.transpose([2, 0, 1]).tolist()

#    centers = []
#    scales = []
    
#    for idx in range(lms.shape[0]):
#        im = Image.open(img_list[idx])
#        w = im.size[0]
#        h = im.size[1]
        # we dont need center !!!!!
        # lsp and lspet dataset doesn't exist groundtruth of center points
#        center_x = (lms[idx][0][lms[idx][0] < w].max() +
#                    lms[idx][0][lms[idx][0] > 0].min()) / 2
#        center_y = (lms[idx][1][lms[idx][1] < h].max() +
#                    lms[idx][1][lms[idx][1] > 0].min()) / 2
#        centers.append([center_x, center_y])
        # ???? +4 what
#        scale = (lms[idx][1][lms[idx][1] < h].max() -
#                lms[idx][1][lms[idx][1] > 0].min() + 4) / 368.0
#        scales.append(scale)

    return kpts
#    return kpts, centers, scales


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


class Sangi_Data(data.Dataset):
    # 1. remove mode parameter
    def __init__(self, root_dir, stride, transforms=None):
        self.img_list = read_sangi_data_file(root_dir)
        #self.kpt_list, self.center_list, self.scale_list = read_sangi_mat_file(root_dir, self.img_list)
        self.kpt_list = read_sangi_mat_file(root_dir, self.img_list)
        self.stride = stride
        self.transforms = transforms
        self.sigma = 3.0


    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = np.array(cv2.imread(img_path), dtype=np.float32)

        kpt = self.kpt_list[index]
        #center = self.center_list[index]
        #scale = self.scale_list[index]

        # data augment and resize
        #img, kpt, center = self.transformer(img,kpt,center,scale) 
        height, width, _ = img.shape
        heatmap = np.zeros((int(height / self.stride), int(width / self.stride), int(len(kpt) + 1)), dtype=np.float32)
        for i in range(len(kpt)):
            # resize from 368 to 46
            x = int(kpt[i][0]) * 1.0 / self.stride
            y = int(kpt[i][1]) * 1.0 / self.stride
            heat_map = guassian_kernel(size_h=height / self.stride, size_w=width / self.stride, center_x=x, center_y=y, sigma=self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i + 1] = heat_map

        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background

        #centermap = np.zeros((height, width, 1), dtype=np.float32)
        #center_map = guassian_kernel(size_h=height, size_w=width, center_x=center[0], center_y=center[1], sigma=3)
        #center_map[center_map > 1] = 1
        #center_map[center_map < 0.0099] = 0
        #centermap[:, :, 0] = center_map

        ToTensor = transforms.ToTensor()
        Normalize = transforms.Normalize([128.0, 128.0, 128.0], [256.0, 256.0, 256.0])
    
        img = ToTensor(img)
        img = Normalize(img)
        #img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0], [256.0, 256.0, 256.0])
        heatmap = ToTensor(heatmap)
        #heatmap = Mytransforms.to_tensor(heatmap)
        #centermap = Mytransforms.to_tensor(centermap)

        return img, heatmap
        #return img, heatmap, centermap

    def __len__(self):
        return len(self.img_list)

