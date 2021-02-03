import torch
import torch.nn.functional as F
import torchvision
import random
import numpy as np
import cv2
import skimage.transform as trans



"""1. Load image and landmarks"""
image = cv2.imread("data/1.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, c = image.shape
src = np.genfromtxt("data/1.pts", skip_header=3, skip_footer=1)
# show_image(image, src, 3)



theta = torch.eye(3).unsqueeze(0)

image = torch.zeros([1,1,5,5])

grid = torch.nn.functional.affine_grid(theta[:,:2], image.size())
print(grid[...,0])
print(grid[...,1])