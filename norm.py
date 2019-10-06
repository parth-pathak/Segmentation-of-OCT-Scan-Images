import numpy as np

def n(img):
    img = np.float64(img)
    img /= 255

    return img

def nu(img):
    img *= 255
    img = np.uint8(img)

    return img
