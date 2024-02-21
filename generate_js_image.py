import cv2
import numpy as np

# Input
image_path = 'image_path.png'
output_path = 'output.js'

# Parameters (Important)
# Increasing these will reduce image quality and point count
k_size = 3  # Kernel size for morphological operations.
iterations = 2  # Number of iterations for the applied operations.

epsilon = 1  # Epsilon value for contour approximation accuracy.

# Hyperparameters
# Run optimizers.py to get these optimized for your image
# The preset values are tuned for typical 4-bit image characteristics.
colors = ['#000000', '#282828', '#6f6f6f', '#eaeaea']
opacities = [1, 0.4671, 0.343, 0.3238]
thresholds = [9, 29, 45, 63, 82, 95, 106, 125, 147, 163, 175, 193, 210, 232, 252]
channel_mappings = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 3, 5, 7, 8, 9, 10, 11],
    [2, 3, 6, 7, 8, 10, 12, 13],
    [4, 5, 6, 7, 10, 11, 13, 14],
]

# Code
def get_channels(image, thresholds):
    masks = []

    thresholds = [0] + thresholds

    for i, j in zip(thresholds[:-1], thresholds[1:]):
        masks.append(np.logical_and(i <= image, image < j))

    return [np.logical_or.reduce([masks[j] for j in i]) for i in channel_mappings]

def get_contours(mask, scale):
    mask = mask.astype(np.uint8) * 255
    mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    contours = [cv2.approxPolyDP(contour, epsilon, True) for contour in contours]

    return [contour[:, 0] / scale for contour in contours if len(contour) > 3]

def get_latex(mask, scale):
    xs, ys = [], []

    for contour in get_contours(mask, scale):
        xs += list(map(str, contour[:, 0])) + ['0/0']
        ys += list(map(str, -contour[:, 1])) + ['0/0']

    print('[Info] layer processed, point count:', len(xs))

    if len(xs) > 10000:
        print('[Warning] >10,000')

    return f'''\\\\operatorname{{polygon}}\\\\left(\\\\left(\\\\left[{','.join(xs)}\\\\right],\\\\left[{','.join(ys)}\\\\right]\\\\right)\\\\right)'''

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 0]

channels = get_channels(image, thresholds)

with open(output_path, 'w') as f:
    f.write(
        '''Calc.setState({'version':11,'expressions':{'list':[{'type':'expression','latex':'a','lines':false,'color':'%s','fillOpacity':'%s' },{'type':'expression','latex':'b','lines':false,'color':'%s','fillOpacity':'%s'},{'type':'expression','latex':'c','lines':false,'color':'%s','fillOpacity':'%s'},{'type':'expression','latex':'d','lines':false,'color':'%s','fillOpacity':'%s'},{'type':'folder','id':'5','title':'don\\'t open this folder','collapsed':true},{'type':'expression','folderId':'5','latex':'a=%s','hidden':true},{'type':'expression','folderId':'5','latex':'b=%s','hidden':true},{'type':'expression','folderId':'5','latex':'c=%s','hidden':true},{'type':'expression','folderId':'5','latex':'d=%s','hidden':true},]}})'''
        % (
            colors[0],
            opacities[0],
            colors[1],
            opacities[1],
            colors[2],
            opacities[2],
            colors[3],
            opacities[3],
            get_latex(channels[0], 4), # resolution scale per layer, change them if you know what you are doing
            get_latex(channels[1], 2),
            get_latex(channels[2], 2),
            get_latex(channels[3], 2),
        )
    )
