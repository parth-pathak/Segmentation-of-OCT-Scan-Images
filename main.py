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
from distance2 import fastsweeping
import sys
#import fast_sweeping

t0 = time()
input_img = cv2.imread('Dataset\\healthy_subject1\\61E9F5B0.tif', 0)
t1 = time()
print('Imread:  '+str(t1-t0)+' s')
crop_img = cropImage(input_img)
size = crop_img.shape
crop_img = cv2.resize(crop_img, (size[1]//4, size[0]//4), interpolation = cv2.INTER_AREA)
size = crop_img.shape
t2 = time()
print('Crop:  '+str(t2-t1)+' s')
nl_img = cv2.fastNlMeansDenoising(crop_img,None,10,7,21)
t3 = time()
print('NL-Means: '+str(t3-t2)+' s')
thresh = LAT(nl_img)
t4 = time()
print('LAT: '+str(t4-t3)+' s')
wt = weight.bright2dark(thresh)
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
t6 = time()
print('Padding: '+str(t6-t5)+' s')
D1 = fastmarching(W, (size[0], size[1]+2), s1, s2)
#d = fast_sweeping.signed_distance(np.array(wt), 1.)
#res = cv2.inpaint(thresh,wt,1,cv2.INPAINT_TELEA)
#res = geodesic.detectBoundary(wt)
#D = res[1]
t7 = time()
print('Fast Marching: '+str(t7-t6)+' s')
D2 = fastsweeping(W, (size[0], size[1]+2), s1, s2)
#arr = gradientFlow(D)
#print(arr)
#img = color.color(arr, crop_img)
t8 = time()

print('Fast sweeping: '+str(t8-t7)+' s')
print('Total: '+str(t8-t0)+' s')
print()

plt.subplot(3,2,1),plt.imshow(crop_img, cmap='gray'),plt.title('Crop and Resize'),plt.xticks([])
plt.subplot(3,2,2),plt.imshow(nl_img, cmap='gray'),plt.title('Denoised'),plt.xticks([])
plt.subplot(3,2,3),plt.imshow(thresh, cmap='gray'),plt.title('LAT'),plt.xticks([])
plt.subplot(3,2,4),plt.imshow(wt, cmap='gray'),plt.title('Weighted'),plt.xticks([])
plt.subplot(3,2,5),plt.imshow(D1, cmap='gray'),plt.title('Fast Marching'),plt.xticks([])
plt.subplot(3,2,6),plt.imshow(D2, cmap='gray'),plt.title('Fast Sweeping'),plt.xticks([])

plt.show()
