import math
import numpy as np
import matplotlib.pyplot as plt


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = (inp * 255).astype(np.uint8)
    return inp


def get_N(W, H):
    """N that maps from unnormalized to normalized coordinates"""
    N = np.zeros((3, 3), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N


def get_N_inv(W, H):
    """N that maps from normalized to unnormalized coordinates"""
    # TODO: do this analytically maybe?
    N = get_N(W, H)
    return np.linalg.inv(N)


def cvt_MToTheta(M, w, h):
    """M is shaped (2, 3) and np.ndarray"""
    M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
    M_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    theta = N @ M_aug @ N_inv
    theta = np.linalg.inv(theta)
    return theta[:2, :]


def cvt_ThetaToM(theta, w, h):
    """theta is shaped (2, 3) and np.ndarray"""
    theta_aug = np.concatenate([theta, np.zeros((1, 3))], axis=0)
    theta_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    M = np.linalg.inv(theta_aug)
    M = N_inv @ M @ N
    return M[:2, :]


def rotatepoints(landmarks, center, rot):
    center_coord = np.zeros_like(landmarks)
    center_coord[:, 0] = center[0]
    center_coord[:, 1] = center[1]

    angle = math.radians(rot)

    rot_matrix = np.array(
        [[math.cos(angle), -1 * math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )

    rotated_coords = np.dot((landmarks - center_coord), rot_matrix) + center_coord

    return rotated_coords


def show_image(image, landmarks, i):
    # print(image.shape)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)
    ax.plot(
        landmarks[0:17, 0],
        landmarks[0:17, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        landmarks[17:22, 0],
        landmarks[17:22, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        landmarks[22:27, 0],
        landmarks[22:27, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        landmarks[27:31, 0],
        landmarks[27:31, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        landmarks[31:36, 0],
        landmarks[31:36, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        landmarks[36:42, 0],
        landmarks[36:42, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        landmarks[42:48, 0],
        landmarks[42:48, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        landmarks[48:60, 0],
        landmarks[48:60, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.plot(
        landmarks[60:68, 0],
        landmarks[60:68, 1],
        marker="o",
        markersize=4,
        linestyle="-",
        color="w",
        lw=2,
    )
    ax.axis("off")
    plt.show()
    plt.savefig(f"{i}.png")
