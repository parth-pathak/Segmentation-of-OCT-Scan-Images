import sys
import math
import numpy as np
label = np.array([])
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

    if abs(UH - UV)<=(h/W[i][j]):
        val1 = math.pow(UH,2)+math.pow(UV,2)-math.pow(h/W[i][j],2)
        val2 = math.pow(UH+UV, 2)-(2*val1)
        u = ((UH+UV)/2)+(0.5*math.sqrt(val2))
    else:
        u = min(UH,UV)+(h/W[i][j])
    return u

def smallest_considered(size):
    x = [-1]
    global U
    global label
    m = sys.maxsize
    for i in range(size[0]):
        for j in range(size[1]):
            if label[i][j]==1:
                if x==[-1] or U[i][j]<m:
                    m = U[i][j]
                    x = [i, j]
    return x

def temp(W, size, i, j):
    global U
    global label
    if label[i][j]!=2:
        u = update(W, size, i, j)
        if u<U[i][j]:
            U[i][j] = u
            if label[i][j]==0:
                label[i][j] = 1

def fastmarching(W, size, s1, s2):
    global U
    global label
    label = np.zeros(size)
    U = np.ones(size)*sys.maxsize

    U[s1[0]][s1[1]] = 0
    label[s1[0]][s1[1]] = 2
    for i in range(size[0]):
        for j in range(size[1]):
            if label[i][j]==0:
                u = update(W, size, i, j)
                if u<U[i][j]:
                    U[i][j] = u
                    label[i][j] = 1

    x=smallest_considered(size)
    while(x!= [-1]):
        label[x[0]][x[1]] = 2
        if x[0]>0:
            temp(W, size, x[0]-1, x[1])
            if x[1]>0:
                temp(W, size, x[0]-1,x[1]-1)
            if x[1]<size[1]-1:
                temp(W, size, x[0]-1,x[1]+1)
        if x[0]<size[0]-1:
            temp(W, size, x[0]+1,x[1])
            if x[1]>0:
                temp(W, size, x[0]+1,x[1]-1)
            if x[1]<size[1]-1:
                temp(W, size, x[0]+1,x[1]+1)
        if x[1]>0:
            temp(W, size, x[0],x[1]-1)
        if x[1]<size[1]-1:
            temp(W, size, x[0],x[1]+1)
        x=smallest_considered(size)

    return U
