from os.path import join
import os
import cv2
import argparse
import numpy as np
import vot
import sys
import json

from deep_mosse import DeepMosse

def rect_from_mask(mask):
    '''
    create an axis-aligned rectangle from a given binary mask
    mask in created as a minimal rectangle containing all non-zero pixels
    '''
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_))
    x1 = np.max(np.nonzero(x_))
    y0 = np.min(np.nonzero(y_))
    y1 = np.max(np.nonzero(y_))

    w = x1 - x0 + 1
    h = y1 - y0 + 1
    # return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    return [x0, y0, w, h]

def mask_from_rect(rect, output_sz):
    '''
    create a binary mask from a given rectangle
    rect: axis-aligned rectangle [x0, y0, width, height]
    output_sz: size of the output [width, height]
    '''
    mask = np.zeros((output_sz[1], output_sz[0]), dtype=np.uint8)
    x0 = max(int(round(rect[0])), 0)
    y0 = max(int(round(rect[1])), 0)
    x1 = min(int(round(rect[0] + rect[2])), output_sz[0])
    y1 = min(int(round(rect[1] + rect[3])), output_sz[1])
    mask[y0:y1, x0:x1] = 1
    return mask

CONFIG = '/home/karolina/INZ/MOSSE_fpga_fork/configs/config.json'
with open(CONFIG, 'r') as json_file:
    config = json.load(json_file)

handle = vot.VOT("mask")
selection = handle.region()
selection = rect_from_mask(selection)

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile)
tracker = DeepMosse(image, selection, config)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    region = tracker.track(image)
    region = mask_from_rect(region, region[2:])
    handle.report(region)