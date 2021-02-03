import math
import torch 
import numpy as np
import matplotlib.pyplot as plt


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = (inp * 255).astype(np.uint8)
    return inp




def get_transform(center, scale, output_size, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(output_size[1]) / h
    t[1, 1] = float(output_size[0]) / h
    t[0, 2] = output_size[1] * (-float(center[0]) / h + .5)
    t[1, 2] = output_size[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -output_size[1] / 2
        t_mat[1, 2] = -output_size[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform_keypoints(kps, meta, invert=False):
    keypoints = kps.copy()
    if invert:
        meta = np.linalg.inv(meta)
    keypoints[:, :2] = np.dot(keypoints[:, :2], meta[:2, :2].T) + meta[:2, 2]
    return keypoints


def kps2box(keypoints):
    x = min(keypoints[:, 0])
    y = min(keypoints[:, 1])
    w = max(keypoints[:, 0]) - x
    h = max(keypoints[:, 1]) - y
    return np.array([x, y, h, w], dtype=np.float32)


def show_image(image, keypoints):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)
    ax.plot(
        keypoints[0:17, 0],
        keypoints[0:17, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        keypoints[17:22, 0],
        keypoints[17:22, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        keypoints[22:27, 0],
        keypoints[22:27, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        keypoints[27:31, 0],
        keypoints[27:31, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        keypoints[31:36, 0],
        keypoints[31:36, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        keypoints[36:42, 0],
        keypoints[36:42, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        keypoints[42:48, 0],
        keypoints[42:48, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        keypoints[48:60, 0],
        keypoints[48:60, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        keypoints[60:68, 0],
        keypoints[60:68, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.axis("off")
    plt.show()

