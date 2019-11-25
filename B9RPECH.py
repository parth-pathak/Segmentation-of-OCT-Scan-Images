import cv2
import math
import numpy as np
import weight
from enhance import LAT
from distance2 import fastsweeping
from flow import gradientFlow

def detect(is_os, nl_img, size):
    rc, gc, bc = cv2.split(is_os)
    for j in range(size[1]):
        for i in range(size[0]):
            if rc[i][j]>0 or gc[i][j]>0 or bc[i][j]>0:
                break
            nl_img[i][j] = 0
    
    wt = weight.dark2bright(nl_img)
    
    W = np.ones((size[0], size[1]+2))
    for i in range(size[0]):
        for j in range(1, size[1]+1):
            W[i][j] = wt[i][j-1]
    s1 = [0, 0]
    s2 = [size[0]-1, size[1]+1]

    D = fastsweeping(W, (size[0], size[1]+2), s1, s2)
    res = gradientFlow(D)

    res = res[:,1:size[1]-1]
    return res
