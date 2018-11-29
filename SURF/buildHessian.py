import numpy as np
from getResponse import getResponse


def buildHessian(r,c,t,m,b):
    v = getResponse(m, r, c, t)
    dxx = getResponse(m, r, c + 1,t) + getResponse(m, r, c - 1,t) - 2 * v
    dyy = getResponse(m, r +1,c,t) + getResponse(m,r-1,c,t)-2*v
    dss = getResponse(t,r,c,0) + getResponse(b,r,c,t) - 2 * v
    dxy = (getResponse(m, r + 1,c+1,t) -  getResponse(m,r+1,c-1,t) - getResponse(m,r-1,c+1,t)+getResponse(m,r-1,c-1,t))/4
    dxs = (getResponse(t,r,c+1,0)- getResponse(t,r,c-1,0)- getResponse(b,r,c+1,t)+getResponse(b,r,c-1,t))/4
    dys = (getResponse(t,r+1,c,0)- getResponse(t,r-1,c,0) - getResponse(b,r+1,c,t) + getResponse(b,r-1,c,t))/4
    H = np.zeros((3,3))
    H[0][0] = dxx
    H[0][1] = dxy
    H[0][2] = dxs
    H[1][0] = dxy
    H[1][1] = dyy
    H[1][2] = dys
    H[2][0] = dxs
    H[2][1] = dys
    H[2][2] = dss
    return H
