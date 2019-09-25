import cv2
import numpy as np
from matplotlib import pyplot as plt
from crop import cropImage
from weightFunc import calculateWeight
from time import time

input_img = cv2.imread('Dataset\\AMD_patient1\\88E4C0.tif', 0)
t1 = time()
img = cropImage(input_img)
t2 = time()
size = img.shape
img = cv2.resize(img, (size[1]*2,size[0]*2), interpolation = cv2.INTER_AREA)
t3 = time()

nl_img = cv2.fastNlMeansDenoising(img,None,10,7,21)
t4 = time()
gaussian_img = cv2.GaussianBlur(img,(7,7), 75)
t5 = time()
bilateral_img = cv2.bilateralFilter(img, 7, 75, 75)
t6 = time()

wt = calculateWeight(img)
t7 = time()

#mask = np.reciprocal(wt)
#res = cv2.inpaint(img,np.uint8(mask),1,cv2.INPAINT_TELEA)
print('Crop time: '+str(t2-t1))
print('Resize time: '+str(t3-t2))
print('NLMeans time: '+str(t4-t3))
print('Gaussian time: '+str(t5-t4))
print('Bilateral time: '+str(t6-t5))
print('Weighting time: '+str(t7-t6))

plt.subplot(2,2,1),plt.imshow(img, cmap='gray'),plt.title('Cropped & resized'),plt.xticks([])
plt.subplot(2,2,2),plt.imshow(nl_img, cmap='gray'),plt.title('Non local means'),plt.xticks([])
plt.subplot(2,2,3),plt.imshow(gaussian_img, cmap='gray'),plt.title('Gaussian'),plt.xticks([])
plt.subplot(2,2,4),plt.imshow(bilateral_img, cmap='gray'),plt.title('Bilateral'),plt.xticks([])
 
plt.show()
