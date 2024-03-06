import os
import cv2
import numpy as np
from tqdm import tqdm
from converters import simplify_image, map_channels, get_latex

video_path = "image_path.mp4"
output_path = "output.js"
start_frame = 0
frame_count = 248
max_points_count = 350
# aim for frame_count * max_points_count < 800000 for 4gb ram limit (local)
# aim for frame_count * max_points_count < 90000 for 5mb file limit (upload)

colors, opacities, values, channel_mappings = (
    ["#000000", "#282828", "#6f6f6f", "#eaeaea"],
    [1, 0.4671, 0.343, 0.3238],
    [0, 19, 37, 54, 72, 88, 100, 115, 136, 155, 169, 184, 201, 221, 242, 255],
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 3, 5, 7, 8, 9, 10, 11],
        [2, 3, 6, 7, 8, 10, 12, 13],
        [4, 5, 6, 7, 10, 11, 13, 14],
    ],
)

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

with open(output_path, "w") as f:
    f.write(
        """Calc.setState({'version':11,'expressions':{'ticker':{'handlerLatex':'p','minStepLatex':'%f','open':true },'list':[{'type':'expression','lines':false,'color':'%s','fillOpacity':'%f','latex':'\\\\left[%s\\\\right]\\\\left[k\\\\right]',},{'type':'expression','lines':false,'color':'%s','fillOpacity':'%f','latex':'\\\\left[%s\\\\right]\\\\left[k\\\\right]',},{'type':'expression','lines':false,'color':'%s','fillOpacity':'%f','latex':'\\\\left[%s\\\\right]\\\\left[k\\\\right]',},{'type':'expression','lines':false,'color':'%s','fillOpacity':'%f','latex':'\\\\left[%s\\\\right]\\\\left[k\\\\right]',},{'type':'expression','latex':'k=1'},{'type':'expression','latex':'p=k\\\\to \\\\left\\\\{k=%d:1,k+1\\\\right\\\\}',},{'collapsed':true,'type':'folder','id':'7'},"""
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

    for i in tqdm(range(frame_count), desc="frames"):
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)[:, :, 0]

        padded = cv2.copyMakeBorder(frame, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        simplified = simplify_image(padded, np.array(values), False)[1:-1, 1:-1]
        channels = map_channels(simplified, channel_mappings)
        latexs = []
        for channel in channels:
            latexs.append(get_latex(channel, max_points_count))

        f.write(
            """{'type':'expression','folderId':'7','latex':'a_{%d}={%s}','hidden':true},{'type':'expression','folderId':'7','latex':'b_{%d}={%s}','hidden':true},{'type':'expression','folderId':'7','latex':'c_{%d}={%s}','hidden':true},{'type':'expression','folderId':'7','latex':'d_{%d}=%s','hidden':true},"""
            % (
                i,
                latexs[0],
                i,
                latexs[1],
                i,
                latexs[2],
                i,
                latexs[3],
            ),
        )
    cap.release()
    f.write("]}})")

print("output file saved as:", os.path.abspath(output_path))
