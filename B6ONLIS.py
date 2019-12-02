import cv2
import math
import numpy as np
import weight
from enhance import LAT
from distance2 import fastsweeping
from flow import gradientFlow

def detect(is_os, ilm, img, size):
    temp = np.zeros(size)
    for j in range(size[1]):
        for i in range(size[0]):
            if ilm[i][j]>0:
                break
            temp[i][j] = img[i][j]
    for j in range(size[1]):
        for i in range(size[0]-1, 0, -1):
            if is_os[i][j]>0:
                break
            temp[i][j] = img[i][j]
    
    wt = weight.bright2dark(temp)
    
    W = np.ones((size[0], size[1]+2))
    for i in range(size[0]):
        for j in range(1, size[1]+1):
            W[i][j] = wt[i][j-1]
    s1 = [0, 0]
    s2 = [size[0]-1, size[1]+1]

    D = fastsweeping(W, (size[0], size[1]+2), s1, s2)
    res = gradientFlow(D)

    return res
