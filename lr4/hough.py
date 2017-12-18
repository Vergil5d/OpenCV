from __future__ import print_function
from sys import argv
import os.path
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
import pdb
import sys
from miptcv_utils import *

def gradient_img(img):
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)
    return magnitude


def hough_transform(img, theta, rho):
    '''
        return: ht_map, thetas, rhos
    '''
    theta_res = np.linspace(-90.0, 0.0, np.ceil(90.0/theta) + 1.0)
    theta_res = np.concatenate((theta_res, -theta_res[len(theta_res)-2::-1]))

    D = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    q = np.ceil(D/rho)
    nrho = 2*q + 1
    rho_res = np.linspace(-q*rho, q*rho, nrho)
    ht_map = np.zeros((len(rho_res), len(theta_res)))
    print('hough map shape  = {}'.format(ht_map.shape))
    for rowIdx in range(img.shape[0]):
        for colIdx in range(img.shape[1]):
            if img[rowIdx, colIdx]:
                for thIdx in range(len(theta_res)):
                    rhoVal = colIdx*np.cos(theta_res[thIdx]*np.pi/180.0) + rowIdx*np.sin(theta_res[thIdx]*np.pi/180)
                    rhoIdx = np.nonzero(np.abs(rho_res-rhoVal) == np.min(np.abs(rho_res-rhoVal)))[0]
                    ht_map[rhoIdx[0], thIdx] += 1
    return ht_map, theta_res, rho_res   
    # thetas = np.arange(0, np.pi / 2, theta)
    # max_abs_rho = 2 * np.sqrt((img.shape[0]**2 + img.shape[1]**2))
    # rhos = np.arange(-max_abs_rho, max_abs_rho, rho)
    # ht_map = np.zeros((thetas.shape[0], rhos.shape[0]), dtype='float32')
    # print('hough map shape  = {}'.format(ht_map.shape))
    # for y in range(img.shape[0]):
    #     for x in range(img.shape[1]):
    #         for theta_numb in range(0, thetas.shape[0]):
    #             cur_theta = thetas[theta_numb]
    #             r = x * np.cos(cur_theta) + y * np.sin(cur_theta)
    #             q_r = int(round((r) / rho, 0)) + rhos.shape[0] // 2
    #             if abs(r - rhos[q_r]) > rho:
    #                 print('problems with R discretization {} {}'.format(r, rhos[q_r]))
    #             if 0 <= q_r < rhos.shape[0]:
    #                 ht_map[theta_numb][q_r] += img[y][x]
    # return ht_map, thetas, rhos



def get_lines(ht_map, n_lines, thetas, rhos, min_delta_rho, min_delta_theta):
    
if __name__ == '__main__':
    assert len(argv) == 9
    src_path, dst_ht_path, dst_lines_path, theta, rho,\
        n_lines, min_delta_rho, min_delta_theta = argv[1:]

    theta = float(theta)
    rho = float(rho)
    n_lines = int(n_lines)
    min_delta_rho = float(min_delta_rho)
    min_delta_theta = float(min_delta_theta)

    assert theta > 0.0
    assert rho > 0.0
    assert n_lines > 0
    assert min_delta_rho > 0.0
    assert min_delta_theta > 0.0

    image = cv2.imread(src_path, 0)
    assert image is not None

    image = image.astype(float)
    gradient = gradient_img(image)

    ht_map, thetas, rhos = hough_transform(gradient, theta, rho)
    cv2.imwrite(dst_ht_path, ht_map)

    lines = get_lines(ht_map, thetas, rhos, n_lines, min_delta_rho, min_delta_theta)
    with open(dst_lines_path, 'w') as fout:
        for line in lines:
            fout.write('%0.3f, %0.3f\n' % line)