import torch
import torch.nn.functional as F
import torchvision
import random
import numpy as np
import cv2
import skimage.transform as trans
from utils import convert_image_np, normalize_transforms, rotatepoints, show_image


"""1. Load image and landmarks"""
image = cv2.imread('1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h,w,c = image.shape
src = np.genfromtxt('1.pts', skip_header=3, skip_footer=1)
show_image(image,src)

"""2. augmentation(scale, rotation, translation)"""
rot = random.randint(-50,50)
dst = rotatepoints(src, [w/2, h/2], rot)
left = min(dst[:,0])
right = max(dst[:,0])
top = min(dst[:,1])
bot = max(dst[:,1])
dst -= [left, top]
dst *= [w/(right-left), h/(bot-top)]
scale = random.uniform(0.8, 1.0)
dst *= scale
dx = random.uniform(-0.05, 0.05) * w
dy = random.uniform(-0.05, 0.05) * h
dst += [dx, dy]

"""3.Warp image to cv_img using OpenCv"""
tr = trans.estimate_transform('affine', src=src, dst=dst)
M = tr.params[0:2,:]
cv_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
show_image(cv_img, dst)

"""4.Affine Transformation Matrix to theta"""
param = np.linalg.inv(tr.params)
theta = normalize_transforms(param[0:2,:], w, h)


"""5.Warp image to tensor_img using grid_sample"""
to_tensor = torchvision.transforms.ToTensor()
tensor_img = to_tensor(image).unsqueeze(0)
theta = torch.Tensor(theta).unsqueeze(0)

grid = F.affine_grid(theta, tensor_img.size())
tensor_img = F.grid_sample(tensor_img, grid)
tensor_img = tensor_img.squeeze(0)
warp_img = convert_image_np(tensor_img)
show_image(warp_img, dst)
