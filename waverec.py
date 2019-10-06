import numpy as np
import pywt
import cv2
import norm

def wr(img, mode='haar', level=1):
    img =  norm.n(img)
    coeffs = pywt.wavedec2(img, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    
    img_H = pywt.waverec2(coeffs_H, mode);
    img_H = norm.nu(img_H)

    return img_H
