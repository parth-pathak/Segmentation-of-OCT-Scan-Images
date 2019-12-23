import cv2
import numpy as np
from matplotlib import pyplot as plt
from crop2 import cropImage
import weight
import color
import points
from time import time
import concurrent.futures
import multiprocessing as mp
import B7ISOS
import B9RPECH
import B8OSRPE
import B1ILM
import B3IPLINL
import B6ONLIS
import B4INLOPL
import B5OPLONL
import B2RNFL
import os

def segment(nl_img, points):
    print(points)
    size = nl_img.shape
    
    d2b = weight.bright2dark(nl_img)
    b2d = np.subtract(1, d2b)
    
    is_os = B7ISOS.detect(nl_img, size, points)
    rpe_ch = B9RPECH.detect(is_os, nl_img, b2d, size, points)
    ilm = B1ILM.detect(is_os, nl_img, d2b, size, points)
    onl_is = B6ONLIS.detect(is_os, nl_img, d2b, size,  points)
    inl_opl = B4INLOPL.detect(ilm, onl_is, nl_img, d2b, size, points)
    opl_onl = B5OPLONL.detect(inl_opl, onl_is, nl_img, b2d, size, points)
    os_rpe = B8OSRPE.detect(is_os, rpe_ch, nl_img, b2d, size, points)
    ipl_inl = B3IPLINL.detect(inl_opl, nl_img, b2d, size, points)
    rnfl = B2RNFL.detect(ilm, ipl_inl, nl_img, b2d, size, points)
    res = color.overlap(nl_img, size, is_os, ilm, onl_is, inl_opl, rpe_ch, opl_onl, os_rpe, ipl_inl, rnfl)
    '''
    emp = np.zeros(size)
    res = color.overlap(nl_img, size, is_os, ilm, onl_is, emp, rpe_ch, emp, os_rpe, emp, emp)
    '''

    return res

img_path = 'H:\\Original Image'
target_path = 'Results'
f = input('Enter file name: ')
print(f, end="...")
t0 = time()
input_img = cv2.imread(os.path.join(img_path, f), 0)
size = input_img.shape
#input_img = cv2.resize(input_img, (size[1]*2, size[0]*2), interpolation = cv2.INTER_AREA)
#size = input_img.shape
print(size)
t2 = time()
crop_img = cropImage(input_img)
t3 = time()

if crop_img.shape[0]==0:
    print('Crop failed')
else:
    nl_img = cv2.fastNlMeansDenoising(crop_img,None,10,7,21)
    part1 = nl_img[:,:size[1]//2]
    part2 = nl_img[:,size[1]//2:]
    
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(segment, part1)
        future2 = executor.submit(segment, part2)
        res1 = future1.result()
        res2 = future2.result()
    with mp.Pool() as p:
        temp = p.map(segment, (part1, part2, ))
        res1 = temp[0]
        res2 = temp[1]
    """

    res1 = segment(part1, points.find(part1))
    '''
    res2 = segment(part2, points.find(part2))
    res = list()
    for i in range(crop_img.shape[0]):
        array = list()
        for j in range(size[1]//2):
            array.append(res1[i][j])
        for j in range(size[1]//2, size[1]):
            array.append(res2[i][j-size[1]//2])
        res.append(array)
    res = np.array(res)
    '''
    t1 = time()
    print(str(t1-t0)+' s')
    print('Crop: '+str(t3-t2)+' s')
    plt.subplot(2,2,1),plt.imshow(input_img, cmap='gray'),plt.title('Original')
    plt.subplot(2,2,2),plt.imshow(crop_img, cmap='gray'),plt.title('Cropped')
    plt.subplot(2,2,3),plt.imshow(res1, cmap='gray'),plt.title('All Layers')
    plt.subplot(2,2,4),plt.imshow(nl_img, cmap='gray'),plt.title('Denoised')
    plt.show()
    #cv2.imwrite(os.path.join(target_path, f), res)


