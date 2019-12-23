def focus(img):
    pixels = []
    size = img.shape
    arr = []
    for i in range(size[0]):
        if sum(img[i])/size[1]<=250:
            for j in range(size[1]):
                arr.append(img[i][j])
    c = sorted(arr)[int(0.98*len(arr))]
    for i in range(size[0]):
        if sum(img[i])/size[1]<=250:
            for j in range(size[1]):
                if img[i][j]>=c:
                    pixels.append(i)
    n = size[0]*size[1]
    pixels = list(set(pixels))
    l = len(pixels)
    a = sum(pixels)/l
    return a

def find(img):
    size = img.shape
    first = img[:size[0]//2,:]
    second = img[size[0]//2:,:]
    p1 = focus(first)
    p2 = focus(second)
    if p1<p2:
        return [size[1]+1, 0]
    else:
        return [0, size[1]+1]
