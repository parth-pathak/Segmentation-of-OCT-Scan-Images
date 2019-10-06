def cropImage(img):
    size = img.shape

    low = 0
    high = (size[0]-1)//2
    prev = 0
    while low < high:
        mid = (low+high)//2
        flag = 0
        a = 0
        b = size[1]-1
        while a<b:
            if img[mid][a]>=100 or img[mid][b]>=100:
                flag = 1
                break
            a += 1
            b -= 1
        if flag==1:
            high = mid
        else:
            low = mid
        if prev==mid:
            break
        else:
            prev = mid
    low -= 20
    if low<0:
        low = 0
    idxt = low

    low = 0
    high = size[0]-1
    prev = 0
    while low < high:
        mid = (low+high)//2
        flag = 0
        a = 0
        b = size[1]-1
        while a<b:
            if img[mid][a]>=100 or img[mid][b]>=100:
                flag = 1
                break
            a += 1
            b -= 1
        if flag==1:
            low = mid
        else:
            high = mid
        if prev==mid:
            break
        else:
            prev = mid
    high += 20
    if high>=size[0]:
        high = size[0]-1
    idxb = high
    
    return img[idxt:idxb,:]
