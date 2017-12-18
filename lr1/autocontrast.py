from __future__ import print_function
from sys import argv
import os.path
import cv2
from matplotlib import pyplot as plt
import numpy as np

import sys
from miptcv_utils import *


def autocontrast(src_path, dst_path, white_perc, black_perc):
    img = cv2.imread(src_path,cv2.IMREAD_GRAYSCALE)
 
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    squre = img.shape[0] * img.shape[1]
    cur_sum = 0

    transform = np.empty(256, dtype=np.uint8)
    #white pixel
    i = 255
    while cur_sum < squre * white_perc:
        cur_sum += hist[i]
        transform[i] = 255
        i -= 1
    white_border = i

    #black pixel
    cur_sum = 0
    i = 0
    while cur_sum < squre * black_perc:
        cur_sum += hist[i]
        transform[i] = 0
        i += 1
    black_border = i
    
    #another pixel
    for i in range(black_border+1, white_border):
        transform[i] = int(float(i - black_border - 1)/(white_border - black_border  - 1) * (255))

    # cum_masked = np.ma.masked_less_equal(cum_sum, black_perc * cum_sum.max()) 
    # cum_masked_max = np.ma.masked_greater_equal(cum_masked, (1.0 - white_perc) * cum_sum.max())
 
    # black_perc_min = cum_masked_max.argmin() #balck pixel
    # white_perc_max = cum_masked_max.argmax() #white pixel
    
    # tr = np.empty(256, dtype='int32')
    # for i in range(0, 256):
    #     if i < black_perc_min:
    #         tr[i] = 0
    #     elif i > white_perc_max:
    #         tr[i] = 255
    #     else:
    #         tr[i] = (i - black_perc_min) * 255 / (white_perc_max - black_perc_min) #another pixel
 
    img_res = transform[img]
    cv2.imwrite(dst_path,img_res)


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = float(argv[3])
    argv[4] = float(argv[4])

    assert 0 <= argv[3] < 1
    assert 0 <= argv[4] < 1

    autocontrast(*argv[1:])
