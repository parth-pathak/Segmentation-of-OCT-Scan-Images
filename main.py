import cv2
import numpy as np
from matplotlib import pyplot as plt
from crop import cropImage
from weightFunc import calculateWeight
from waverec import wr
from enhance import LAT
from time import time
import fast_sweeping
import os
import norm
import tkinter
import matplotlib
t0 = time()
matplotlib.use('TkAgg')

root = os.path.dirname(__file__)
img_path = os.path.join(os.path.expanduser(root), 'Dataset', 'AMD_patient1', '1098670.tif')

input_img = cv2.imread(img_path, 0)
t1 = time()
crop_img = cropImage(input_img)
t2 = time()
nl_img = cv2.fastNlMeansDenoising(crop_img,None,15,7,21)
t3 = time()
thresh = LAT(nl_img)
t4 = time()
wt = calculateWeight(thresh)
t5 = time()
d = fast_sweeping.signed_distance(np.array(norm.n(wt)), 0.003921569)
t6 = time()
#res = cv2.inpaint(thresh,wt,1,cv2.INPAINT_TELEA)

print('%20s:%1.20f s'%('Image read', t1-t0))
print('%20s:%1.20f s'%('Crop', t2-t1))
print('%20s:%1.20f s'%('Denoising', t3-t2))
print('%20s:%1.20f s'%('LAT', t4-t3))
print('%20s:%1.20f s'%('Weight', t5-t4))
print('%20s:%1.20f s'%('Fast-sweeping', t6-t5))
print('%20s:%1.20f s'%('Total', t6-t0))

plt.subplot(2,2,1),plt.imshow(nl_img, cmap='gray'),plt.title('Crop and Denoise'),plt.xticks([])
plt.subplot(2,2,2),plt.imshow(thresh, cmap='gray'),plt.title('LAT'),plt.xticks([])
plt.subplot(2,2,3),plt.imshow(wt, cmap='gray'),plt.title('Weighted'),plt.xticks([])
plt.subplot(2,2,4),plt.imshow(d, cmap='gray'),plt.title('Fast-sweeping'),plt.xticks([])

plt.show()
