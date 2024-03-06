import os
import cv2
import numpy as np
from tqdm import tqdm
from optimizers import get_optimized
from converters import simplify_image, map_channels, get_latex

image_path = 'image_path.png'
output_path = 'output.js'
iterations = 64 # how many iterations to optimize the hyperparameters

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.uint8)

colors, opacities, values, channel_mappings = get_optimized(image, iterations)

# colors, opacities, values, channel_mappings = ['#000000', '#282828', '#6f6f6f', '#eaeaea'], [1, 0.4671, 0.343, 0.3238], [0, 19, 37, 54, 72, 88, 100, 115, 136, 155, 169, 184, 201, 221, 242, 255], [[0, 1, 2, 3, 4, 5, 6, 7], [1, 3, 5, 7, 8, 9, 10, 11], [2, 3, 6, 7, 8, 10, 12, 13], [4, 5, 6, 7, 10, 11, 13, 14]]

padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
simplified = simplify_image(padded, np.array(values))[1:-1, 1:-1]
channels = map_channels(simplified, channel_mappings)
latexs = []
for i in tqdm(channels, 'calculating contours'):
    latexs.append(get_latex(i))

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
            latexs[0],
            latexs[1],
            latexs[2],
            latexs[3],
        )
    )

print('output file saved as:', os.path.abspath(output_path))