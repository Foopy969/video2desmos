# video2desmos
A simple script to convert videos or images into desmos graphs using polygons

# Usage
## Install dependancies
Run
```py
pip install -r requirements.txt
```
## Run script
Edit the parameters in `image2desmos.py`
```py
image_path = 'image_path_here.png'
output_path = 'output_path_here.js'
iterations = 64 # how many iterations to optimize the hyperparameters (layer colors and opacities)
```
Run `image2desmos.py`
## Copy to `desmos.com`
### Chrome instructions
1.  f12
2.  type `allow pasting`
3.  drag `output.js` onto console
4.  enter and wait
