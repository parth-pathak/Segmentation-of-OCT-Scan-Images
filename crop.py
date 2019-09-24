def cropImage(img):
    size = img.shape

    idxt = 0
    flag = 0
    while idxt < size[0]:
        a = 0
        b = size[1]-1
        while a<b:
            if img[idxt][a]>=150 or img[idxt][b]>=150:
                flag = 1
                break
            a += 1
            b -= 1
        if flag==1:
            break
        idxt += 1
    idxt -= 20
    if idxt<0:
        idxt = 0
    
    idxb = size[0]-1
    flag = 0
    while idxb >= 0:
        a = 0
        b = size[1]-1
        while a<b:
            if img[idxb][a]>=100 or img[idxb][b]>=100:
                flag = 1
                break
            a += 1
            b -= 1
        if flag==1:
            break
        idxb -= 1
    idxb += 20
    if idxb >= size[0]:
        idxb = size[0]
    
    return img[idxt:idxb,:]
