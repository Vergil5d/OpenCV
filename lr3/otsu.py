from __future__ import print_function
from sys import argv
import os.path
import cv2
from matplotlib import pyplot as plt
import numpy as np


def otsu(src_path, dst_path):
    img = cv2.imread(src_path,0)
    out_img = img.copy().astype('int32')
    img_w, img_h = img.shape[:2]
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    sum_all = 0
    for i in range(256):
        sum_all += i * hist[i]
    sumB, wB, wF, var_max, between, mB, mF, threshold = 0, 0, 0, 0, 0,0,0,0
    total = img_h*img_w
    for i in range(256):
        wB += hist[i]
        if (wB == 0): continue
        wF = total - wB
        if (wF == 0) : break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        between = wB * wF * (mB - mF)**2 
        if (between > var_max):
            var_max = between
            threshold = i
    for x in range(img_w): 
        for y in range(img_h): 
            if img[x, y] >= threshold: 
                out_img[x, y] = 255
            else:
                out_img[x, y] = 0
    # print ("threshold is:", threshold)
    cv2.imwrite(dst_path, out_img)

if __name__ == '__main__':
    assert len(argv) == 3
    assert os.path.exists(argv[1])
    otsu(*argv[1:])
