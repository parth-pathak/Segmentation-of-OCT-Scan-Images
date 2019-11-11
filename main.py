import cv2
import numpy as np
from matplotlib import pyplot as plt
from crop import cropImage
from detection import detectEdge
import color
from weightFunc import calculateWeight
from waverec import wr
from enhance import LAT
from time import time
import geodesic
import sys
#import fast_sweeping

print(sys.getrecursionlimit())
t0 = time()
input_img = cv2.imread('Dataset\\AMD_patient2\\90C683A0.tif', 0)
t1 = time()
print('Imread:  '+str(t1-t0)+' s')
crop_img = cropImage(input_img)
size = crop_img.shape
crop_img = cv2.resize(crop_img, (size[1]//10, size[0]//10), interpolation = cv2.INTER_AREA)
size = crop_img.shape
sys.setrecursionlimit(max(1000, size[0]*size[1] + 1))
print(sys.getrecursionlimit())
t2 = time()
print('Crop:  '+str(t2-t1)+' s')
nl_img = cv2.fastNlMeansDenoising(crop_img,None,10,7,21)
t3 = time()
print('NL-Means: '+str(t3-t2)+' s')
thresh = LAT(crop_img)
t4 = time()
print('thresh: '+str(t4-t3)+' s')
wt = calculateWeight(thresh)
#idx = detectEdge(thresh)
#wt = createEdge(crop_img, thresh, idx)
t5 = time()
print('Weight: '+str(t5-t4)+' s')
#d = fast_sweeping.signed_distance(np.array(wt), 1.)
#res = cv2.inpaint(thresh,wt,1,cv2.INPAINT_TELEA)
res = geodesic.detectBoundary(wt)
#img = color.color(res, crop_img)
img = np.zeros((size[0], size[1]+2))
b = [size[0]-1,size[1]+1]
#while b!=[0,0]:
#    img[b[0]][b[1]] = 255
#    b = res[0][b[0]][b[1]]
t6 = time()

print('Boundary: '+str(t6-t5)+' s')
print('Total: '+str(t6-t0)+' s')
print()

plt.subplot(2,2,1),plt.imshow(crop_img, cmap='gray'),plt.title('Crop'),plt.xticks([])
plt.subplot(2,2,2),plt.imshow(thresh, cmap='gray'),plt.title('LAT'),plt.xticks([])
plt.subplot(2,2,3),plt.imshow(res[1], cmap='gray'),plt.title('Geodesic'),plt.xticks([])
plt.subplot(2,2,4),plt.imshow(img, cmap='gray'),plt.title('Boundary'),plt.xticks([])

plt.show()
