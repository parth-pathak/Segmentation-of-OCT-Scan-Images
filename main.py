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
import B8OSRPE
import B1ILM
import B3IPLINL
import B6ONLIS
import B4INLOPL
import B5OPLONL
import B2RNFL
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

d2b = weight.bright2dark(nl_img)
b2d = np.subtract(1, d2b)
t4 = time()

is_os = B7ISOS.detect(nl_img, size)
t5 = time()

rpe_ch = B9RPECH.detect(is_os, nl_img, b2d, size)
t6 = time()

ilm = B1ILM.detect(is_os, nl_img, d2b, size)
t7 = time()

onl_is = B6ONLIS.detect(is_os, nl_img, d2b, size)
t8 = time()

inl_opl = B4INLOPL.detect(ilm, onl_is, nl_img, d2b, size)
t9 = time()

opl_onl = B5OPLONL.detect(inl_opl, onl_is, nl_img, b2d, size)
t10 = time()

os_rpe = B8OSRPE.detect(is_os, rpe_ch, nl_img, b2d, size)
t11 = time()

ipl_inl = B3IPLINL.detect(inl_opl, nl_img, b2d, size)
t12 = time()

rnfl = B2RNFL.detect(ilm, ipl_inl, nl_img, b2d, size)
t13 = time()

res = color.overlap(nl_img, size, is_os, ilm, onl_is, inl_opl, rpe_ch, opl_onl, os_rpe, ipl_inl, rnfl)
t14 = time()

print('Imread:  '+str(t1-t0)+' s')
print('Crop:  '+str(t2-t1)+' s')
print('NL-Means: '+str(t3-t2)+' s')
print('Weight: '+str(t4-t3)+' s')
print('B7: '+str(t5-t4)+' s')
print('B9: '+str(t6-t5)+' s')
print('B1: '+str(t7-t6)+' s')
print('B6: '+str(t8-t7)+' s')
print('B4: '+str(t9-t8)+' s')
print('B5: '+str(t10-t9)+' s')
print('B8: '+str(t11-t10)+' s')
print('B3: '+str(t12-t11)+' s')
print('B2: '+str(t13-t12)+' s')
print('Overlap: '+str(t14-t13)+' s')

print('Total: '+str(t14-t0)+' s')
print()

plt.subplot(2,1,1),plt.imshow(crop_img, cmap='gray')
plt.subplot(2,1,2),plt.imshow(res)
plt.show()
