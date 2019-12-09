import sys
import math
import numpy as np

U = np.array([])

def update(W, size, i, j):
    h = 1
    global U
    if i==0:
        UH = U[i+1][j]
    elif i==size[0]-1:
        UH = U[i-1][j]
    else:
        UH = min(U[i-1][j], U[i+1][j])
    if j==0:
        UV = U[i][j+1]
    elif j==size[1]-1:
        UV = U[i][j-1]
    else:
        UV = min(U[i][j-1], U[i][j+1])
    if W[i][j]==0:
        W[i][j] += 0.001

    if abs(UH - UV)<(h/W[i][j]):
        val1 = math.pow(UH,2)+math.pow(UV,2)-math.pow(h/W[i][j],2)
        val2 = (2*math.pow(h/W[i][j], 2))-math.pow(UH-UV, 2)
        u = ((UH+UV)/2)+(0.5*math.sqrt(val2))
    else:
        u = min(UH,UV)+(h/W[i][j])
    return u

def fastsweeping(W, size, s1, s2):
    global U
    label = np.zeros(size)
    U = np.ones(size)*sys.maxsize

    U[s1[0]][s1[1]] = 0
    label[s1[0]][s1[1]] = 1
    
    for i in range(size[0]):
        for j in range(size[1]):
            if label[i][j]==0:
                U[i][j] = min(U[i][j], update(W, size, i, j))
    for i in range(size[0]):
        for j in range(size[1]-1, -1, -1):
            if label[i][j]==0:
                U[i][j] = min(U[i][j], update(W, size, i, j))
    for i in range(size[0]-1, -1, -1):
        for j in range(size[1]-1, -1, -1):
            if label[i][j]==0:
                U[i][j] = min(U[i][j], update(W, size, i, j))
    for i in range(size[0]-1, -1, -1):
        for j in range(size[1]):
            if label[i][j]==0:
                U[i][j] = min(U[i][j], update(W, size, i, j))
    
    return U
