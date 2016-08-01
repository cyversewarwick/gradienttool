import numpy as np

def inflectionPoints(y):
    d = np.diff(y) > 0
    out = []
    dc = d[0]
    for i in range(1,len(d)):
        if dc == d[i]:
            continue
        out.append((i,[-1,1][dc]))
        dc = d[i]
    return np.array(out,dtype=np.int)
