import numpy as np
import norm
import multiprocessing as mp
res = []
pi = []
wt = 0
def adj(u, w):
    l = []
    size = w.shape
    if u[0]>0:
        l.append([u[0]-1, u[1]])
        if u[1]>0:
            l.append([u[0]-1,u[1]-1])
        if u[1]<size[1]-1:
            l.append([u[0]-1,u[1]+1])
    if u[0]<size[0]-1:
        l.append([u[0]+1,u[1]])
        if u[1]>0:
            l.append([u[0]+1,u[1]-1])
        if u[1]<size[1]-1:
            l.append([u[0]+1,u[1]+1])
    if u[1]>0:
        l.append([u[0],u[1]-1])
    if u[1]<size[1]-1:
        l.append([u[0],u[1]+1])
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            x1 = l[i][0]
            x2 = l[j][0]
            y1 = l[i][1]
            y2 = l[j][1]
            if w[x1][y1]>w[x2][y2]:
                temp = l[i]
                l[i] = l[j]
                l[j] = temp
    return l

def graphDFS(s1, s2, w, adjList):
    global res
    global pi
    global wt
    wt += w[s1[0]][s1[1]]
    res[s1[0]][s1[1]] = wt
    if s1!=s2:
        for v in adjList[s1[0]][s1[1]]:
            if pi[s1[0]][s1[1]] != v:
                if wt+w[v[0]][v[1]]<res[v[0]][v[1]] or res[v[0]][v[1]]==0:
                    pi[v[0]][v[1]] = s1
                    graphDFS(v, s2, w, adjList)
    wt -= w[s1[0]][s1[1]]

def detectBoundary(wt):
    global pi
    global res
    global adjList
    size = wt.shape
    W = np.ones((size[0], size[1]+2))*255
    W[:, 1:-1] = wt
    W = np.subtract(255, W)
    
    s1 = [0,0]
    s2 = [size[0]-1,size[1]+1]

    color = np.multiply(255, np.ones((size[0], size[1]+2)))
    res = np.zeros((size[0], size[1]+2))
    pi = [[0 for a in range(size[1]+2)] for b in range(size[0])]
    adjList = [[] for x in range(size[0])]
    for x in range(size[0]):
        for y in range(size[1]+2):
            adjList[x].append(adj([x,y], W))
    print('DFS initiated..')
    graphDFS(s1, s2, W, adjList)
    print('DFS finished..')
    return [pi,res]
