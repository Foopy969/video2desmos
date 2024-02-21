import cv2
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

# This is optional
# This script is for optimizing your layer values and opacities for a specific image
# Copy the results to the start of the image script (replace whats already there)

# Inputs
image_path = 'image_path.png'
iterations = 200

# Code
np.set_printoptions(suppress=True)

def get_thresholds(image, depth):
    threshold, _ = cv2.threshold(image ,0 ,255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if depth:
        yield from get_thresholds(image[image <= threshold], depth - 1)
        yield from get_thresholds(image[threshold < image], depth - 1)
    yield threshold

def objective(variables):
    a_o, a_v, b_o, b_v, c_o, c_v, d_o, d_v = variables

    A = (1 - a_o) + a_o * a_v
    B = (1 - b_o) + b_o * b_v
    C = (1 - c_o) + c_o * c_v
    D = (1 - d_o) + d_o * d_v
    AB = (1 - b_o) * A + b_o * b_v
    AC = (1 - c_o) * A + c_o * c_v
    AD = (1 - d_o) * A + d_o * d_v
    BC = (1 - c_o) * B + c_o * c_v
    BD = (1 - d_o) * B + d_o * d_v
    CD = (1 - d_o) * C + d_o * d_v
    ABC = (1 - c_o) * AB + c_o * c_v
    ABD = (1 - d_o) * AB + d_o * d_v
    ACD = (1 - d_o) * AC + d_o * d_v
    BCD = (1 - d_o) * BC + d_o * d_v
    ABCD = (1 - d_o) * ABC + d_o * d_v

    e = np.array([A, B, C, D, AB, AC, AD, BC, BD, CD, ABC, ABD, ACD, BCD, ABCD])
    diff = np.sort(e) - np.sort(v)
    return np.sum(diff**2)

def get_sorted(variables):
    a_o, a_v, b_o, b_v, c_o, c_v, d_o, d_v = variables

    A = (1 - a_o) + a_o * a_v
    B = (1 - b_o) + b_o * b_v
    C = (1 - c_o) + c_o * c_v
    D = (1 - d_o) + d_o * d_v
    AB = (1 - b_o) * A + b_o * b_v
    AC = (1 - c_o) * A + c_o * c_v
    AD = (1 - d_o) * A + d_o * d_v
    BC = (1 - c_o) * B + c_o * c_v
    BD = (1 - d_o) * B + d_o * d_v
    CD = (1 - d_o) * C + d_o * d_v
    ABC = (1 - c_o) * AB + c_o * c_v
    ABD = (1 - d_o) * AB + d_o * d_v
    ACD = (1 - d_o) * AC + d_o * d_v
    BCD = (1 - d_o) * BC + d_o * d_v
    ABCD = (1 - d_o) * ABC + d_o * d_v

    equation_names = {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "AB": AB,
        "AC": AC,
        "AD": AD,
        "BC": BC,
        "BD": BD,
        "CD": CD,
        "ABC": ABC,
        "ABD": ABD,
        "ACD": ACD,
        "BCD": BCD,
        "ABCD": ABCD,
    }

    result = sorted(equation_names.items(), key=lambda x: x[1])
    return [x for x, _ in result], [i for _, i in result]

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 0]

thresholds = np.sort([0] + [*get_thresholds(image, 3)])
v = (thresholds[:-1] + thresholds[1:]) / 510

bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]

min_fun = 99
min_x = []

for i in tqdm(range(iterations)):
    initial_guess = np.random.rand(8)
    result = minimize(objective, initial_guess, bounds=bounds, method='COBYLA')

    if result.fun < min_fun:
        min_fun = result.fun
        min_x = result.x

names, values = get_sorted(min_x)
values = np.array(values + [1])

a_o, a_v, b_o, b_v, c_o, c_v, d_o, d_v = min_x

print('colors =', ['#' + hex(int(i * 255))[2:].zfill(2) * 3 for i in [a_v, b_v, c_v, d_v]])
print('opacities =', [a_o, b_o, c_o, d_o])
print('thresholds =', list(np.round((values[1:] + values[:-1]) * 255 / 2).astype(np.uint8)))
print('channel_mappings =', [[j for j, x in enumerate(names) if i in x] for i in "ABCD"])