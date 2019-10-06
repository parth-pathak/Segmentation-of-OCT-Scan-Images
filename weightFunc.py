import cv2
import numpy as np
import norm

def calculateWeight(img):
    size = img.shape
    
    sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3)
    dx = norm.n(sobelx)
    dy = norm.n(np.absolute(sobely))
    
    wt = np.multiply(-10, np.subtract(1, dx))
    for i in range(size[0]):
        for j in range(size[1]):
            wt[i][j] = wt[i][j]*dy[i][j]
    wt = np.subtract(1, np.exp(wt))

    wt = norm.nu(wt)
    return wt
