import cv2
import numpy as np
from detection import detectEdge
from enhance import LAT
count = 0
def createEdge(img, thresh, idx):
    global count
    count += 1
    size = img.shape
    rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    if count==10 or size[0]<4:
        return rgb
    rc, gc, bc = cv2.split(rgb)
    x = idx[1]
    y = idx[0]
    rc[y][x] = 255
    rc[y-1][x] = 255
    rc[y-2][x] = 255
    rc[y-3][x] = 255
    rc[y-4][x] = 255
    minimum = y
    while x>-1:
        x -= 1
        while thresh[y-1][x]>=50 and y>=0:
            y -= 1
        while thresh[y][x]<50 and y<size[0]-1:
            y += 1
        rc[y][x] = 255
        rc[y-1][x] = 255
        rc[y-2][x] = 255
        rc[y-3][x] = 255
        rc[y-4][x] = 255
        if y>minimum:
            minimum = y
    x = idx[1]
    y = idx[0]
    while x<size[1]-1:
        x += 1
        while thresh[y-1][x]>=50 and y>=0:
            y -= 1
        while thresh[y][x]<50 and y<size[0]-1:
            y += 1
        rc[y][x] = 255
        rc[y-1][x] = 255
        rc[y-2][x] = 255
        rc[y-3][x] = 255
        rc[y-4][x] = 255
        if y>minimum:
            minimum = y
    img_rgba = cv2.merge((rc, gc, bc))
    print(idx)
    print(minimum)
    thresh2 = LAT(img[minimum:,:])
    index = detectEdge(thresh2)
    wt = createEdge(img[minimum:,:], thresh2, index)
    res = np.concatenate((img_rgba[:minimum,:,:], wt), axis=0)
    return res
