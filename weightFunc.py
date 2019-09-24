import cv2
import numpy as np

def calculateWeight(img):
    sobely = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=9)
    sobelx = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=9)
    
    size = img.shape
    dx = []
    dy = []
    for i in range(size[0]):
        rowx = []
        rowy = []
        for j in range(size[1]):
            rowx.append(sobelx[i][j]*img[i][j])
            rowy.append(sobely[i][j]*img[i][j])
        dx.append(rowx)
        dy.append(rowy)
    
    dx = np.array(dx)
    dy = np.absolute(np.array(dy))
    cv2.normalize(dx, dx, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
    cv2.normalize(dy, dy, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
    
    wt = np.multiply(-10, np.subtract(1, dx))
    for i in range(size[0]):
        for j in range(size[1]):
            wt[i][j] = wt[i][j]*dy[i][j]
    wt = np.subtract(1, np.exp(wt))

    return wt
