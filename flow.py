import cv2
import math
import random
import numpy as np
gamma = []
def gradient(D, size):
    global gamma
    g = [0, 0]
    i = gamma[0]
    j = gamma[1]
    """
    if i-math.floor(i)>0.5:
        i = math.ceil(i)
    else:
        i = math.floor(i)
    if j-math.floor(j)>0.5:
        j = math.ceil(j)
    else:
        j = math.floor(j)
    """
    
    if i==size[0]-1:
        g[0] = D[i][j] - D[i-1][j]
    elif i==0:
        g[0] = D[i+1][j] - D[i][j]
    else:
        g[0] = D[i+1][j] - D[i-1][j]
    
    if j==size[1]-1:
        g[1] = D[i][j] - D[i][j-1]
    elif j==0:
        g[1] = D[i][j+1] - D[i][j]
    else:
        g[1] = D[i][j+1] - D[i][j-1]

    if g==[0,0]:
        g = [0,1]
    p = math.sqrt(sum([math.pow(x,2) for x in g]))
    p += 1
    for k in range(len(g)):
        g[k] /= p
    
    return g

def gradientFlow(D):
    global gamma
    size = D.shape
    res = []
    gamma = [size[0]-1, size[1]-1]
    s1 = [0, 0]
    h = 0.8
    rcs = np.zeros(size)
    while gamma!=s1:
        rcs[gamma[0]][gamma[1]] = 255
        G = gradient(D, size)
        gamma[0] = gamma[0] - (h*G[0])
        gamma[1] = gamma[1] - (h*G[1])
        gamma[0] = math.floor(gamma[0])
        gamma[1] = math.floor(gamma[1])
        if gamma[0]<0:
            gamma[0] = 0
        if gamma[1]<0:
            gamma[1] = 0
    rcs = rcs[:,1:size[1]-1]
    return rcs
