from __future__ import print_function
from sys import argv
import os.path

import cv2
import numpy as np
from miptcv_utils import *

def gamma_correction(src_path, dst_path, a, b):
    original = cv2.imread(src_path, cv2.IMREAD_COLOR)
    cur = original / 255.0
    cur = np.power(cur, b)
    cur = cur * a *255
    cv2.imwrite(dst_path, cur)

if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = float(argv[3])
    argv[4] = float(argv[4])

    gamma_correction(*argv[1:])