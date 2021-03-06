import cv2
import math
import numpy as np
import weight
from enhance import LAT
from distance2 import fastsweeping
from flow import gradientFlow

def detect(inl_opl, img, wt, size, points):
    flag = 0
    temp = np.zeros(size)
    for j in range(size[1]):
        for i in range(size[0]):
            if i+5<size[0] and inl_opl[i+5][j]>0:
                flag = 1
            if inl_opl[i][j]>0:
                flag = 0
            temp[i][j] = flag*wt[i][j]
    
    W = np.ones((size[0], size[1]+2))
    for i in range(size[0]):
        for j in range(1, size[1]+1):
            W[i][j] = temp[i][j-1]
    s1 = [0, points[0]]
    s2 = [size[0]-1, points[1]]

    D = fastsweeping(W, (size[0], size[1]+2), s1, s2)
    res = gradientFlow(D, s1, s2)

    return res
