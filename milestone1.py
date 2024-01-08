import numpy as np
import cv2
import random
from skimage.transform import radon, iradon
import matplotlib.pyplot as plt


def add_sp_noise(img, prob):
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]

    return output


def reconstruction_and_projection(binarized_img, num_of_picture):
    theta_value_list = []
    rme_list = []

    for theta_max_value in range(10, 180, 10):
        theta = np.linspace(0, 180, theta_max_value)
        r = radon(binarized_img, theta)
        ir = iradon(r, theta)
        _, bir = cv2.threshold(ir, 0, 1, cv2.THRESH_BINARY)

        rme = np.count_nonzero(binarized_img - bir) / np.count_nonzero(binarized_img)
        rme_list.append(rme)
        theta_value_list.append(theta_max_value)

        print(f'RMS reconstruction error: {rme} for theta:{theta_max_value}')

    print(f'Smallest RMS reconstruction error: {min(rme_list)}, '
          f'for theta:{theta_value_list[rme_list.index(min(rme_list))]}')

    # Plotting
    bar_width = 0.45
    index = range(len(theta_value_list))

    plt.figure(figsize=(10, 6))
    plt.bar(index, rme_list, width=bar_width)

    plt.xlabel('Projection number')
    plt.ylabel('RMS reconstruction error')
    plt.grid(axis='y', alpha=0.5, color='gray', linestyle='-', linewidth=1)
    plt.title(f' Picture {num_of_picture} RMS reconstruction error for increasing projection angle')
    plt.xticks([i + bar_width / 2 for i in index], theta_value_list)
    plt.show()


def reconstruction_and_noise(binary_img, num_of_picture):
    reconstruction_error_list = []
    noise_probability_list = []

    for prob in np.arange(0.05, 0.9, 0.1):
        noisy_img_binary = add_sp_noise(binary_img, prob)
        theta = np.linspace(0., 180., max(binary_img.shape))
        noisy_r = radon(noisy_img_binary, theta=theta)
        noisy_ir = iradon(noisy_r, theta=theta)
        _, noisy_bir = cv2.threshold(noisy_ir, 0, 1, cv2.THRESH_BINARY)

        error = np.count_nonzero(binary_img - noisy_bir) / np.count_nonzero(binary_img)

        reconstruction_error_list.append(error)
        noise_probability_list.append("{:.2f}".format(prob))

        print(f'RMS reconstruction error {error} with noise probability {prob}')

    bar_width = 0.45
    index = range(len(noise_probability_list))

    plt.figure(figsize=(10, 6))
    plt.bar(index, reconstruction_error_list, width=bar_width, color='green')

    plt.xlabel('Noise probability')
    plt.ylabel('RMS Reconstruction error')
    plt.grid(axis='y', alpha=0.5, color='gray', linestyle='-', linewidth=1)
    plt.title(f' Picture {num_of_picture} RMS reconstruction error for increasing noise probability')
    plt.xticks([i + bar_width / 2 for i in index], noise_probability_list)
    plt.show()


# ----------------- Main ------------------
for pic in range(1, 6):
    image = cv2.imread(f"pictures/test_{pic}.png", 0)
    _, BI = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

    reconstruction_and_projection(BI, pic)
    reconstruction_and_noise(BI, pic)
