import sys
import numpy as np
from geodesic import adj

def minDistance(size, dist, sptSet):
    min = sys.maxsize

    min_index = [-1, -1]
    for i in range(size[0]):
        for j in range(size[1]):
            if dist[i][j] < min and sptSet[i][j] == False:
                min = dist[i][j]
                min_index = [i, j]

    return min_index

def getDistanceMap(img):
    size = img.shape
    
    adjList = []
    for i in range(size[0]):
        l = []
        for j in range(size[1]):
            l.append(adj([i,j], img))
        adjList.append(l)
    
    dist = [[sys.maxsize] * size[1]] * size[0]
    dist[0][0] = img[0][0]
    sptSet = [[False] * size[1]] * size[0]

    for a in range(size[0]):
        for b in range(size[1]):
            u = minDistance(size, dist, sptSet)
            sptSet[u[0]][u[1]] = True

            for v in adjList[u[0]][u[1]]:
                if sptSet[v[0]][v[1]] == False and dist[v[0]][v[1]] > dist[u[0]][u[1]] + img[v[0]][v[1]]: 
                    dist[v[0]][v[1]] = dist[u[0]][u[1]] + img[v[0]][v[1]]

    return [dist,sptSet]
