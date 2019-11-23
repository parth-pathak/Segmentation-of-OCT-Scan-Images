import cv2
import numpy as np
from matplotlib import pyplot as plt
from crop import cropImage
from detection import detectEdge
import color
import weight
from enhance import LAT
from flow import gradientFlow
from time import time
import geodesic
from distance import fastmarching
import sys
#import fast_sweeping

print(sys.getrecursionlimit())
t0 = time()
input_img = cv2.imread('Dataset\\AMD_patient2\\90C683A0.tif', 0)
t1 = time()
print('Imread:  '+str(t1-t0)+' s')
crop_img = cropImage(input_img)
size = crop_img.shape
#crop_img = cv2.resize(crop_img, (size[1]//8, size[0]//8), interpolation = cv2.INTER_AREA)
size = crop_img.shape
sys.setrecursionlimit(max(1000, size[0]*size[1] + 1))
print(sys.getrecursionlimit())
t2 = time()
print('Crop:  '+str(t2-t1)+' s')
nl_img = cv2.fastNlMeansDenoising(crop_img,None,10,7,21)
t3 = time()
print('NL-Means: '+str(t3-t2)+' s')
thresh = LAT(nl_img)
t4 = time()
print('thresh: '+str(t4-t3)+' s')
wt = weight.dark2bright(thresh)
#idx = detectEdge(thresh)
#wt = createEdge(crop_img, thresh, idx)
t5 = time()
print('Weight: '+str(t5-t4)+' s')
W = np.ones((size[0], size[1]+2))
for i in range(size[0]):
    for j in range(1, size[1]+1):
        W[i][j] = wt[i][j-1]
s1 = [0, 0]
s2 = [size[0]-1, size[1]+1]
D = fastmarching(W, (size[0], size[1]+2), s1, s2)
#d = fast_sweeping.signed_distance(np.array(wt), 1.)
#res = cv2.inpaint(thresh,wt,1,cv2.INPAINT_TELEA)
#res = geodesic.detectBoundary(wt)
#D = res[1]
t6 = time()
print('Distance Map: '+str(t6-t5)+' s')
#arr = gradientFlow(D)
#print(arr)
#img = color.color(arr, crop_img)
t7 = time()

print('Gradient Descent: '+str(t7-t6)+' s')
print('Total: '+str(t7-t0)+' s')
print()

plt.subplot(2,2,1),plt.imshow(nl_img, cmap='gray'),plt.title('Crop & Denoised'),plt.xticks([])
plt.subplot(2,2,2),plt.imshow(thresh, cmap='gray'),plt.title('LAT'),plt.xticks([])
plt.subplot(2,2,3),plt.imshow(wt, cmap='gray'),plt.title('Weight'),plt.xticks([])
plt.subplot(2,2,4),plt.imshow(D, cmap='gray'),plt.title('Geodesic'),plt.xticks([])

plt.show()
