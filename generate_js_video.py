import cv2
import numpy as np
from tqdm import tqdm

# Input
video_path = "video_path.mp4"
output_path = "output.js"
start_frame = 2048
frame_count = 512

# Parameters (Important)
# Increasing these will reduce image quality and point count
k_size = 3  # Kernel size for morphological operations.
iterations = 5  # Number of iterations for the applied operations.

epsilon = 5  # Epsilon value for contour approximation accuracy.

# Hyperparameters
# Run optimizers.py to get these optimized for your image
# The preset values are tuned for typical 4-bit image characteristics.
colors = ["#000000", "#282828", "#6f6f6f", "#eaeaea"]
opacities = [1, 0.4671, 0.343, 0.3238]
thresholds = [9, 29, 45, 63, 82, 95, 106, 125, 147, 163, 175, 193, 210, 232, 252]
channel_mappings = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 3, 5, 7, 8, 9, 10, 11],
    [2, 3, 6, 7, 8, 10, 12, 13],
    [4, 5, 6, 7, 10, 11, 13, 14],
]


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
        xs += list(map(str, contour[:, 0])) + ["0/0"]
        ys += list(map(str, -contour[:, 1])) + ["0/0"]

    if len(xs) > 10000:
        print("[Info] layer processed, point count:", len(xs))
        print("[Warning] >10,000")

    return f"""\\\\operatorname{{polygon}}\\\\left(\\\\left(\\\\left[{','.join(xs)}\\\\right],\\\\left[{','.join(ys)}\\\\right]\\\\right)\\\\right)"""


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

with open(output_path, "w") as f:
    f.write('''Calc.setState({'version':11,'expressions':{'ticker':{'handlerLatex':'p','minStepLatex':'%f','open':true },'list':[{'type':'expression','lines':false,'color':'%s','fillOpacity':'%f','latex':'\\\\left[%s\\\\right]\\\\left[k\\\\right]',},{'type':'expression','lines':false,'color':'%s','fillOpacity':'%f','latex':'\\\\left[%s\\\\right]\\\\left[k\\\\right]',},{'type':'expression','lines':false,'color':'%s','fillOpacity':'%f','latex':'\\\\left[%s\\\\right]\\\\left[k\\\\right]',},{'type':'expression','lines':false,'color':'%s','fillOpacity':'%f','latex':'\\\\left[%s\\\\right]\\\\left[k\\\\right]',},{'type':'expression','latex':'k=1'},{'type':'expression','latex':'p=k\\\\to \\\\left\\\\{k=%d:1,k+1\\\\right\\\\}',},{'collapsed':true,'type':'folder','id':'7'},'''
        % (
            1000 / cap.get(cv2.CAP_PROP_FPS),
            colors[0],
            opacities[0],
            ",".join([f"a_{{{i}}}" for i in range(frame_count)]),
            colors[1],
            opacities[1],
            ",".join([f"b_{{{i}}}" for i in range(frame_count)]),
            colors[2],
            opacities[2],
            ",".join([f"c_{{{i}}}" for i in range(frame_count)]),
            colors[3],
            opacities[3],
            ",".join([f"d_{{{i}}}" for i in range(frame_count)]),
            frame_count,
        )
    )

    for i in tqdm(range(frame_count)):
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)[:, :, 0]

        channels = get_channels(frame, thresholds)

        f.write('''{'type':'expression','folderId':'7','latex':'a_{%d}={%s}','hidden':true},{'type':'expression','folderId':'7','latex':'b_{%d}={%s}','hidden':true},{'type':'expression','folderId':'7','latex':'c_{%d}={%s}','hidden':true},{'type':'expression','folderId':'7','latex':'d_{%d}=%s','hidden':true},'''
            % (
                i,
                get_latex(channels[0], 4), # resolution scale per layer, change them if you know what you are doing
                i,
                get_latex(channels[1], 2),
                i,
                get_latex(channels[2], 2),
                i,
                get_latex(channels[3], 2),
            ),
        )
    cap.release()
    f.write(']}})')
