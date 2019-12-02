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
import B1ILM
import B6ONLIS
import sys

t0 = time()
input_img = cv2.imread('Dataset\\AMD_patient1\\FF6684D0.tif', 0)
t1 = time()

crop_img = cropImage(input_img)
size = crop_img.shape
#crop_img = cv2.resize(crop_img, (size[1]//2, size[0]//2), interpolation = cv2.INTER_AREA)
#size = crop_img.shape
t2 = time()

nl_img = cv2.fastNlMeansDenoising(crop_img,None,10,7,21)
t3 = time()

is_os = B7ISOS.detect(nl_img, size)
t4 = time()

ilm = B1ILM.detect(is_os, nl_img, size)
t5 = time()

res = color.overlap(nl_img, size, is_os, ilm)
t6 = time()

print('Imread:  '+str(t1-t0)+' s')
print('Crop:  '+str(t2-t1)+' s')
print('NL-Means: '+str(t3-t2)+' s')
print('B7: '+str(t4-t3)+' s')
print('B1: '+str(t5-t4)+' s')
print('Overlap: '+str(t6-t5)+' s')

print('Total: '+str(t6-t0)+' s')
print()

plt.subplot(2,2,1),plt.imshow(input_img,  cmap='gray'),plt.title('Input Image'),plt.xticks([])
plt.subplot(2,2,2),plt.imshow(crop_img, cmap='gray'),plt.title('Crop and Resize'),plt.xticks([])
plt.subplot(2,2,3),plt.imshow(nl_img, cmap='gray'),plt.title('NL Means'),plt.xticks([])
plt.subplot(2,2,4),plt.imshow(res),plt.title('Result'),plt.xticks([])

plt.show()
