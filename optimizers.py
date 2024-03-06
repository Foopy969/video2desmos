import cv2
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

method = "COBYLA"

def get_thresholds(image, depth):
    threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if depth:
        yield from get_thresholds(image[image <= threshold], depth - 1)
        yield from get_thresholds(image[threshold < image], depth - 1)
    yield threshold


def objective(variables, v):
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


def get_optimized(image, iterations):
    thresholds = np.sort([0] + [*get_thresholds(image, 3)])
    v = (thresholds[:-1] + thresholds[1:]) / 510

    bounds = [(0, 1), (0, 0.25), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0.75, 1)]

    min_fun = 99
    min_x = []

    for _ in tqdm(range(iterations), desc='optimizing hyperparameters'):
        initial_guess = np.random.rand(8)
        result = minimize(
            objective, initial_guess, args=v, bounds=bounds, method=method
        )

        if result.fun < min_fun:
            min_fun = result.fun
            min_x = result.x

    names, values = get_sorted(min_x)
    values = np.array(values + [1])

    a_o, a_v, b_o, b_v, c_o, c_v, d_o, d_v = min_x

    return (
        ["#" + hex(int(i * 255))[2:].zfill(2) * 3 for i in [a_v, b_v, c_v, d_v]],
        [a_o, b_o, c_o, d_o],
        values * 255,
        [[j for j, x in enumerate(names) if i in x] for i in "ABCD"],
    )
