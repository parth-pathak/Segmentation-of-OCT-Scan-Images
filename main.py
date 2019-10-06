import cv2
import numpy as np
from matplotlib import pyplot as plt
from crop import cropImage
from weightFunc import calculateWeight
from waverec import wr
from enhance import LAT
from time import time

input_img = cv2.imread('Dataset\\AMD_patient1\\88E4C0.tif', 0)
t1 = time()
crop_img = cropImage(input_img)
t2 = time()
nl_img = cv2.fastNlMeansDenoising(crop_img,None,15,7,21)
t3 = time()
thresh = LAT(nl_img)
t4 = time()
wt = calculateWeight(thresh)
t5 = time()
#res = cv2.inpaint(thresh,wt,1,cv2.INPAINT_TELEA)

print('Crop:  '+str(t2-t1)+' s')
print('Denoising: '+str(t3-t2)+' s')
print('LAT: '+str(t4-t3)+' s')
print('Weight: '+str(t5-t4)+' s')

plt.subplot(2,2,1),plt.imshow(nl_img, cmap='gray'),plt.title('Crop and Denoise'),plt.xticks([])
plt.subplot(2,2,2),plt.imshow(thresh, cmap='gray'),plt.title('LAT'),plt.xticks([])
plt.subplot(2,2,3),plt.imshow(wt, cmap='gray'),plt.title('Weighted'),plt.xticks([])

plt.show()
