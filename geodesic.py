def choosePoints(wt):
    size = wt.shape
    W = np.ones((size[0], size[1]+2))
    W[:, 1:-1] = wt
    
    s1 = W[0,0]
    s2 = W[-1,-1]

    return (s1, s2)
