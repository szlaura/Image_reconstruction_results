import numpy as np
import cv2
from skimage.transform import radon, iradon
import scipy.optimize as opt
import matplotlib.pyplot as plt


# ||Ax-b||^2 + g(x) -> min
def objective_func_calculate(x, matrix, b):
    term1 = np.linalg.norm(np.dot(matrix, x) - b) ** 2
    return term1


def annealing(bir_ravel, r_ravel, bounds):
    res = opt.dual_annealing(objective_func_calculate, bounds=bounds, args=(bir_ravel, r_ravel), maxiter=100)
    print(f'{res.x} RMS reconstruction error: {res.fun}, {res.message}')
    return res


# ----------------- Main ------------------
for pic in range(1, 6):
    theta_value_list = []
    rme_list = []
    best_annealing = 1000
    best_theta = 0

    image = cv2.imread(f"pictures/test_{pic}.png", cv2.IMREAD_GRAYSCALE)
    print(f"test_{pic}.png Simulated annealing:")

    initial_guess = image.flatten()
    bounds_for_annealing = [(0, 1)] * len(initial_guess)

    _, bi_pic = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)

    for theta_max_value in range(10, 180, 10):
        actual_rme = 0
        theta = np.linspace(0, theta_max_value, max(bi_pic.shape))
        r = radon(bi_pic, theta)
        ir = iradon(r, theta)
        _, bir = cv2.threshold(ir, 0, 1, cv2.THRESH_BINARY)

        res_of_annealing = annealing(bir.ravel(), r.ravel(), bounds_for_annealing)
        if best_annealing > res_of_annealing.fun:
            best_annealing = res_of_annealing.fun
            best_theta = theta_max_value

        actual_rme = res_of_annealing.fun
        rme_list.append(actual_rme)
        theta_value_list.append(theta_max_value)

    # Plotting
    bar_width = 0.45
    index = range(len(theta_value_list))

    plt.figure(figsize=(10, 6))
    plt.bar(index, rme_list, width=bar_width)
    plt.grid(axis='y', alpha=0.5, color='gray', linestyle='-', linewidth=1)

    plt.xlabel('Projection number')
    plt.ylabel('RMS reconstruction error')
    plt.title(f' Picture {pic} RMS reconstruction error for increasing projection angle')
    plt.xticks([i + bar_width / 2 for i in index], theta_value_list)
    plt.legend()
    plt.show()
    print(f"For test_{pic}.png the smallest RMS reconstruction error: {best_annealing}"
          f" with max Theta value: {best_theta}\n")
