import numpy as np
import pywt
import cv2

def wr(img, mode='haar', level=1):
    img =  np.float32(img)
    img /= 255
    coeffs = pywt.wavedec2(img, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    
    img_H = pywt.waverec2(coeffs_H, mode);
    img_H *= 255;
    img_H =  np.uint8(img_H)

    return img_H
