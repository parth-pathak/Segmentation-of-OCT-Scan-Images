import cv2
import math
import numpy as np
import weight
from enhance import LAT

def gradientFlow(w, size, s2):
    global gamma
    size = D.shape
    res = []
    gamma = [size[0]-1, size[1]-1]
    s1 = [0, 0]
    h = 0.8
    rc = np.zeros(size)
    gc = np.zeros(size)
    bc = np.zeros(size)
    r = random.randrange(0, 256, 4)
    g = random.randrange(0, 256, 4)
    b = random.randrange(0, 256, 4)
    while gamma!=s1:
        rc[gamma[0]][gamma[1]] = 255
        gc[gamma[0]][gamma[1]] = 255
        bc[gamma[0]][gamma[1]] = 0
        G = gradient(D, size)
        gamma[0] = gamma[0] - (h*G[0])
        gamma[1] = gamma[1] - (h*G[1])
        gamma[0] = math.floor(gamma[0])
        gamma[1] = math.floor(gamma[1])
        if gamma[0]<0:
            gamma[0] = 0
        if gamma[1]<0:
            gamma[1] = 0
    
    img_rgba = cv2.merge((rc, gc, bc))
    return img_rgba

def newMethod(img, size):
    thresh = LAT(nl_img)
    wt = weight.bright2dark(thresh)

    W = np.ones((size[0], size[1]+2))
    for i in range(size[0]):
        for j in range(1, size[1]+1):
            W[i][j] = wt[i][j-1]
    s1 = [0, 0]
    s2 = [size[0]-1, size[1]+1]

    res = gradientFlow(W, size, s2)

    res = res[:,1:size[1]-1]
    return res
