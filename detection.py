def detectEdge(img):
    size = img.shape

    low = 0
    high = (size[0]-1)//2
    prev = 0
    v = 0
    while low < high:
        mid = (low+high)//2
        flag = 0
        a = 0
        b = size[1]-1
        while a<b:
            if img[mid][a]>=100:
                flag = 1
                break
            if img[mid][b]>=100:
                flag = 2
                break
            a += 1
            b -= 1
        if flag==1 or flag==2:
            high = mid
        else:
            low = mid
        if prev==mid:
            if flag==1:
                v = a
            elif flag==2:
                v = b
            break
        else:
            prev = mid
    return (low,v)
