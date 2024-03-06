import cv2
import numpy as np
from numba import jit
from tqdm import tqdm

@jit(nopython=True)
def anti_dither(image, palette):
    image = image.astype(np.int16)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            pixel = image[i][j]
            err = pixel - palette[np.argmin(np.abs(palette - pixel))]
            image[i + 1, j    ] -= err * 0.4375
            image[i - 1, j + 1] -= err * 0.1875
            image[i    , j + 1] -= err * 0.3125
            image[i + 1, j + 1] -= err * 0.0625
    return image

@jit(nopython=True)
def get_palette_index(image, palette):
    indices = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            indices[i, j] = np.argmin(np.abs(palette - image[i, j]))
    return indices

@jit(nopython=True)
def cellular_grow(image):
    updated = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            counts = np.zeros(16)
            counts[image[i    , j    ]] += 3 # current pixel
            counts[image[i - 1, j - 1]] += 1
            counts[image[i - 1, j    ]] += 1
            counts[image[i - 1, j + 1]] += 1
            counts[image[i    , j - 1]] += 1
            counts[image[i    , j + 1]] += 1
            counts[image[i + 1, j - 1]] += 1
            counts[image[i + 1, j    ]] += 1
            counts[image[i + 1, j + 1]] += 1
            updated[i][j] = np.argmax(counts)
    return updated

def simplify_image(image, values, show_progress=True):
    if show_progress:
        bar = tqdm(range(3), 'Simplifying image')
    image = anti_dither(image, values)
    if show_progress:
        bar.update(1)
        bar.refresh()
    indices = get_palette_index(image, values)
    if show_progress:
        bar.update(1)
        bar.refresh()
    indices = cellular_grow(indices)
    if show_progress:
        bar.update(1)
        bar.refresh()
    return indices

def map_channels(image, channel_mappings):
    masks = [(image == i) for i in range(16)]
    return [np.logical_or.reduce([masks[j] for j in i]) for i in channel_mappings]

def get_latex(mask, limit=10000):
    mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    simplified = contours
    epsilon = 0

    while True:
        xs, ys = [], []
        simplified = [contour[:, 0] for contour in simplified if len(contour) > 3]

        for contour in simplified:
            xs += list(map(str, contour[:, 0])) + ["0/0"]
            ys += list(map(str, -contour[:, 1])) + ["0/0"]

        if len(xs) < limit:
            break
        else:
            epsilon += 0.1
            simplified = [cv2.approxPolyDP(contour, epsilon, True) for contour in contours]
            
    return f"""\\\\operatorname{{polygon}}\\\\left(\\\\left(\\\\left[{','.join(xs)}\\\\right],\\\\left[{','.join(ys)}\\\\right]\\\\right)\\\\right)"""