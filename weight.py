import cv2
import numpy as np

def delx(img, size):
    res = np.zeros((size[0], size[1]))
    for i in range(1, size[0]-1):
        for j in range(size[1]):
            res[i][j] = img[i+1][j] - img[i-1][j]
    return res

def dely(img, size):
    res = np.zeros((size[0], size[1]))
    for i in range(size[0]):
        for j in range(1, size[1]-1):
            res[i][j] = img[i][j+1] - img[i][j-1]
    return res

def bright2dark(img):
    size = img.shape
    img = np.float64(img)
    dx = np.add(255, delx(img, size))/510
    dy = np.add(255, np.absolute(dely(img, size)))/510
    
    wt = np.multiply(-10, np.subtract(1, dx))
    for i in range(size[0]):
        for j in range(size[1]):
            wt[i][j] = wt[i][j]*dy[i][j]

    return np.exp(wt)

def dark2bright(img):
    return np.subtract(1, bright2dark(img))
