import math
def gradient(D, size, gamma):
    g = [0, 0]
    
    if gamma[0]==size[0]-1:
        g[0] = D[gamma[0]][gamma[1]] - D[gamma[0]-1][gamma[1]]
    elif gamma[0]==0:
        g[0] = D[gamma[0]+1][gamma[1]] - D[gamma[0]][gamma[1]]
    else:
        g[0] = D[gamma[0]+1][gamma[1]] - D[gamma[0]-1][gamma[1]]
    
    if gamma[1]==size[1]-1:
        g[1] = D[gamma[0]][gamma[1]] - D[gamma[0]][gamma[1]-1]
    elif gamma[1]==0:
        g[1] = D[gamma[0]][gamma[1]+1] - D[gamma[0]][gamma[1]]
    else:
        g[1] = D[gamma[0]][gamma[1]+1] - D[gamma[0]][gamma[1]-1]

    p = math.sqrt(sum([math.pow(x,2) for x in g]))
    if p!=0:
        for i in range(len(g)):
            g[i] /= p
            g[i] = 0-g[i]
    return g

def gradientFlow(D):
    size = D.shape
    res = []
    gamma = [size[0]-1, size[1]-1]
    s1 = [0, 0]
    h = 0.8
    while gamma!=s1:
        print(gamma)
        res.append(gamma)
        G = gradient(D, size, gamma)
        if G[0]>0:
            gamma[0] = math.floor(gamma[0] - (h*G[0]))
        else:
            gamma[0] = math.ceil(gamma[0] - (h*G[0]))
        if G[1]>0:
            gamma[1] = math.floor(gamma[1] - (h*G[1]))
        else:
            gamma[1] = math.ceil(gamma[1] - (h*G[1]))
    return res
