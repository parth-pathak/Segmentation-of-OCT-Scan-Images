import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from crop import cropImage
import color
import weight
from enhance import LAT
from flow import gradientFlow
from time import time
from distance2 import fastsweeping
import B7ISOS
import B9RPECH
import sys

t0 = time()
input_img = cv2.imread('Dataset\\AMD_patient1\\FE4424E0.tif', 0)
t1 = time()

crop_img = cropImage(input_img)
size = crop_img.shape
crop_img = cv2.resize(crop_img, (size[1]//2, size[0]//2), interpolation = cv2.INTER_AREA)
size = crop_img.shape
t2 = time()

nl_img = cv2.fastNlMeansDenoising(crop_img,None,10,7,21)
t3 = time()

is_os = B7ISOS.detect(nl_img, size)
t4 = time()

rgb = cv2.cvtColor(nl_img, cv2.COLOR_GRAY2RGB)
rc, gc, bc = cv2.split(rgb)
r, g, b = cv2.split(is_os)
for i in range(size[0]):
    for j in range(size[1]):
        rc[i][j] = max(rc[i][j], r[i][j])
        gc[i][j] = max(gc[i][j], g[i][j])
        bc[i][j] = max(bc[i][j], b[i][j])
res = cv2.merge((rc, gc, bc))
#rpe_ch = B9RPECH.detect(is_os, nl_img, size)
t5 = time()

print('Imread:  '+str(t1-t0)+' s')
print('Crop:  '+str(t2-t1)+' s')
print('NL-Means: '+str(t3-t2)+' s')
print('B7: '+str(t4-t3)+' s')
print('B9: '+str(t5-t4)+' s')

print('Total: '+str(t5-t0)+' s')
print()

plt.subplot(2,2,1),plt.imshow(crop_img, cmap='gray'),plt.title('Crop and Resize'),plt.xticks([])
plt.subplot(2,2,2),plt.imshow(nl_img, cmap='gray'),plt.title('Denoised'),plt.xticks([])
plt.subplot(2,2,3),plt.imshow(is_os),plt.title('Boundary'),plt.xticks([])
plt.subplot(2,2,4),plt.imshow(res),plt.title('Result'),plt.xticks([])

plt.show()
