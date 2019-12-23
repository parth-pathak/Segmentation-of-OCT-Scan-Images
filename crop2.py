def cropImage(img):
    pixels = []
    size = img.shape
    arr = []
    for i in range(size[0]):
        if sum(img[i])/size[1]<=250:
            for j in range(size[1]):
                arr.append(img[i][j])
    c = sorted(arr)[int(0.95*len(arr))]
    for i in range(size[0]):
        if sum(img[i])/size[1]<=250:
            for j in range(size[1]):
                if img[i][j]>=c:
                    pixels.append(i)
    n = size[0]*size[1]
    l = len(pixels)
    t = int(0.1*l)
    lis = sorted(pixels)
    idxt = pixels[t] - 40
    idxb = pixels[l-t] + 30
    if idxt<0:
        idxt = 0
    if idxb>=size[0]:
        idxb = size[0]-1

    return img[idxt:idxb,:]
