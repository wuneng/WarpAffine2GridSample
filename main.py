import torch
import torch.nn.functional as F
import torchvision
import random
import numpy as np
import cv2
import skimage.transform as trans
import warnings
from utils import get_transform, transform_keypoints, convert_image_np, show_image
warnings.filterwarnings("ignore")


"""1. Load image and landmarks"""
image = cv2.imread("data/image.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_height, image_width, _ = image.shape
keypoints = np.genfromtxt("data/image.pts", skip_header=3, skip_footer=1)
show_image(image, keypoints)
print("Origin image size: (%d, %d)" % (image.shape[1], image.shape[0]))



"""2. augmentation(scale, rotation, translation)"""
rot = random.randint(-50, 50)
center = [image_width / 2., image_height / 2.]
scale = max(image_height, image_width) / 200. * random.uniform(0.9, 1.1)



"""3.Warp image to cv_img using OpenCv"""
output_height = int(image_height * random.uniform(0.9, 1.1))
output_width = int(image_width * random.uniform(0.9, 1.1))
output_size = (output_width, output_height)
meta = get_transform(center, scale, output_size, rot)
cv_img = cv2.warpAffine(image, meta[:2], output_size)
cv_keypoints = transform_keypoints(keypoints, meta)
show_image(cv_img, cv_keypoints)
print("Opencv image size: (%d, %d)" % (cv_img.shape[1], cv_img.shape[0]))


"""4.Affine Transformation Matrix to theta"""
src = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.float32)
dst = transform_keypoints(src, meta)

# normalize to [-1, 1]
src = src / [image_width, image_height] * 2 - 1
dst = dst / [output_width, output_height] * 2 - 1
theta = trans.estimate_transform("affine", src=dst, dst=src).params


"""5.Warp image to tensor_img using grid_sample"""
to_tensor = torchvision.transforms.ToTensor()
tensor = to_tensor(image).unsqueeze(0)
theta = torch.tensor(theta, dtype=torch.float32).unsqueeze(0)
output_size = torch.Size((1, 3, output_height, output_width))
grid = F.affine_grid(theta[:, :2], output_size)
tensor = F.grid_sample(tensor, grid, align_corners=False)
tensor = tensor.squeeze(0)
torch_img = convert_image_np(tensor)
show_image(torch_img, cv_keypoints)
print("Torch image size: (%d, %d)" % (torch_img.shape[1], torch_img.shape[0]))
