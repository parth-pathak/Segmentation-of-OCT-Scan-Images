import numpy as np
import cv2
import norm

WS = 51
C = 0.01
def LAT(img):
    size = img.shape
    img = norm.n(img)
    kernel = np.ones((WS,WS), np.float64)/(WS*WS)
    dst = cv2.filter2D(img, -1, kernel)
    dst = np.subtract(dst, img)
    for i in range(size[0]):
        for j in range(size[1]):
            if dst[i][j] > C:
                img[i][j] = 0
    img = norm.nu(img)
    return img
