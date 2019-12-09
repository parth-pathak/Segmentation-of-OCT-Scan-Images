import cv2
import math
import numpy as np
import weight
from enhance import LAT
from distance2 import fastsweeping
from flow import gradientFlow
from matplotlib import pyplot as plt

def detect(is_os, img, wt, size):
    flag = 0
    temp = np.zeros(size)
    for j in range(size[1]):
        for i in range(size[0]):
            if i+8<size[0] and is_os[i+8][j]>0:
                flag = 1
            if is_os[i][j]>0:
                flag = 0
            temp[i][j] = flag*wt[i][j]
    
    W = np.ones((size[0], size[1]+2))
    for i in range(size[0]):
        for j in range(1, size[1]+1):
            W[i][j] = temp[i][j-1]
    s1 = [0, 0]
    s2 = [size[0]-1, size[1]+1]

    D = fastsweeping(W, (size[0], size[1]+2), s1, s2)
    res = gradientFlow(D)

    return res
