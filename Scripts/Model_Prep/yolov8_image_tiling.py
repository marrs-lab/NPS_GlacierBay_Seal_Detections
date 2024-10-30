# -*- coding: utf-8 -*-
"""
@author: James David Poling III. Duke University Marine Robotics and Remote Sensing Lab

takes pictures and creates crops of 640 x 640 of the original with grid naming
"""

from PIL import Image
from itertools import product
import os


dir_in = r''
dir_out = r''


d = 640  # recommend 640px images. 1280px may work but will be more intense to train


def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size

    grid = product(range(0, h, d), range(0, w, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        print(i, j)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)


for filename in os.listdir(dir_in)[::1]:
    if  os.path.isfile(os.path.join(dir_in,filename)) and (filename.lower().endswith('.jpg') or filename.endswith(".JPG")):
       tile(filename, dir_in, dir_out, d)
