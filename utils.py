import numpy as np
import cv2
import imutils
import torch
from torch.nn import Module
import torchvision.models as models
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import math

from finnmodels import YOLO_finn


def init_seeds(seed=0):
    np.random.seed(seed)


def get_VGG_backbone(version='vgg11', pretrained=True):

    versions = ['vgg11', 'vgg19']
    assert version in versions, '\"{}\" not recognized as vgg version, the possible versions are: {}'.format(version, versions)

    if version == 'vgg11':
        vgg = models.vgg11(pretrained=pretrained, progress=True)
        vgg = vgg.features[:3]
    elif version == 'vgg19':
        vgg = models.vgg19(pretrained=pretrained, progress=True)
        vgg = vgg.features[:5]

    # print('Using VGG features:')
    # print(vgg)

    vgg.eval()

    return vgg


def get_CF_backbone(config_path, weights_path):

    class CFBackbone(Module):

        def __init__(self, conv_features, take_first_n):
            super(CFBackbone, self).__init__()
            self.conv_features = conv_features[:take_first_n]

        def forward(self, x):
            x = 2.0 * x - torch.tensor([1.0], device=x.device)
            for mod in self.conv_features:
                x = mod(x)
            
            return x


    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    detector = YOLO_finn(config).to(torch.device('cpu'))
    checkpoint_dict = torch.load(weights_path, map_location='cpu')
    detector.load_state_dict(checkpoint_dict['model'])

    backbone = detector.backbone
    cf_backbone = CFBackbone(backbone.conv_features, 5)
    cf_backbone.eval()

    return cf_backbone


# pre-processing the image... DEPRECATED
def pre_process(img):
    # print('USING DEPRECATED PREPROCESSING')
    # get the size of the img...
    height, width = img.shape
    img = np.log(img + 1)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    # use the hanning window...
    window = window_func_2d(height, width)
    img = img * window

    return img


def plot_params_search(filepath):

    with open(filepath, "r") as file:
        lines = file.readlines()
    
    lines = [line for line in lines if '[' in line]
    # print(lines)
    # for line in lines:
    #     print(line)

    sigma_lr = [line.split(']')[0].strip('[').split(',') for line in lines]
    sigmas = [float(line[0]) for line in sigma_lr]
    lrs = [float(line[1].strip()) for line in sigma_lr]
    ious = [float(line.split(':')[1].split()[0]) for line in lines]
    iou_dict = {}
    for s, l, i in zip(sigmas, lrs, ious):
        iou_dict[(s, l)] = i
    sigmas = np.unique(sigmas)
    lrs = np.unique(lrs)
    X, Y = np.meshgrid(sigmas, lrs)
    Z = np.zeros_like(X)
    # print(X.shape, Y.shape, Z.shape)
    # print(X)
    # print(Y)
    for x, s in enumerate(sigmas):
        for y, l in enumerate(lrs):
            if (s, l) in iou_dict:
                Z[y, x] = iou_dict[(s, l)]
    # for k, v in iou_dict.items():
    #     print(k, v)
    # print(Z)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('Sigma')
    ax.set_ylabel('lr')
    ax.set_zlabel('iou')
    ax.set_xticks(sigmas)
    ax.set_yticks(lrs)
    plt.savefig('params_search.png')
    plt.show()
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(sigmas, lrs, ious)
    # plt.show()
    

# input is a list of 4 points: [X1, Y1, X2, Y2, X3, Y3, X4, Y4]
# output is [x, y, w, h]
def check_bbox(points):

    Xs = points[::2]    # odd coords
    Ys = points[1::2]   # even coords

    left = min(Xs)
    right = max(Xs)
    top = min(Ys)
    bottom = max(Ys)

    width = right - left
    height = bottom - top

    return [left, top, width, height]

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

def load_gt(gt_file, format='xyxy'):

    with open(gt_file, 'r') as file:
        lines = file.read().splitlines()

    delimiters = [',', '\n']

    for d in delimiters:
        if d in lines[0]:
            lines = [line.split(d) for line in lines]
            break
    # lines = [rect_from_mask([int(coord) for coord in line]) for line in lines]
    lines = [[int(float(coord)) if not math.isnan(float(coord)) else None for coord in line] for line in lines]
    
    # if standard == 'vot2013':
    #     result = lines
    # else:
    #     test_line = lines[0]
    #     result = [[line[0], line[1], line[2]-line[0], line[5]-line[1]] for line in lines]
    result = [check_bbox(line) if not None in line else [None, None, None, None] for line in lines]

    # returns in xywh format
    return result


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # print('boxes shapes:', box1.shape, box2.shape)
 
    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[0], box1[0] + box1[2]
    b1_y1, b1_y2 = box1[1], box1[1] + box1[3]
    b2_x1, b2_x2 = box2[0], box2[0] + box2[2]
    b2_y1, b2_y2 = box2[1], box2[1] + box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1, a_min=0, a_max=None) * np.clip(inter_rect_y2 - inter_rect_y1, a_min=0, a_max=None)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def pad_img(img, padded_size, pad_type='center'):

    if padded_size == 0:
        return img, [0, 0, 0, 0]

    height, width = img.shape
    # print('hw:', height, width)
    assert height <= padded_size and width <= padded_size

    if pad_type == 'topleft':
        result = np.zeros((padded_size, padded_size))
        result[:height, :width] = img
        padding = [0, padded_size-height, 0, padded_size-width]

    elif pad_type == 'center':
        h_diff = padded_size - height
        w_diff = padded_size - width
        if h_diff % 2 == 0:
            top_pad = h_diff / 2
            bottom_pad = top_pad
        else:
            top_pad = h_diff // 2
            bottom_pad = top_pad + 1
        
        if w_diff % 2 == 0:
            left_pad = w_diff / 2
            right_pad = left_pad
        else:
            left_pad = w_diff // 2
            right_pad = left_pad + 1

        padding = [top_pad, bottom_pad, left_pad, right_pad]
        padding = [int(pad) for pad in padding]
        top_pad, bottom_pad, left_pad, right_pad = padding
        # print('padding:', padding)
        result = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT)
    
    # cv2.imshow('padded', result.astype(np.uint8))
    # print(result)

    return result, padding


# used for linear mapping...
def linear_mapping(img):
    return (img - img.min()) / (img.max() - img.min())



def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)

    win = mask_col * mask_row

    return win

# img in [C, H, W] shape
def random_warp(img, i='0'):
    a = -180 / 16
    b = 180 / 16
    r = a + (b - a) * np.random.uniform()

    channels, height, width = img.shape
    img = img.transpose(1, 2, 0)
    img_rot = imutils.rotate_bound(img, r)
    img_resized = cv2.resize(img_rot, (width, height))
    # cv2.imshow(i+' sample', img_resized)
    if channels == 1:
        img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = img_resized.transpose(2, 0, 1)
    # rotate the image...
    # matrix_rot = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), r, 1)
    # img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
    # img_rot = img_rot.astype(np.float32) / 255
    return img_resized


