import numpy as np
import numpy.matlib as matlib
import math
from HaarX import  HaarX
from HaarY import HaarY
def getDescriptor(ip,bUpright, bExtended, img, verbose):
    X = round(ip["x"])
    Y = round(ip["y"])
    S = round (ip["scale"])

    if (bUpright):
        co = 1
        si = 0
    else:
        co = np.cos(ip["orientation"])
        si = np.sin(ip["orientation "])

    [lb,kb] = np.mgrid[-4:4,-4:4]
    lb=np.reshape(lb,(-1,1))
    kb=np.reshape(kb,(-1,1))
    [jl,il] = np.mgrid[0:3,0:3]
    jl=np.reshape(jl,(-1,1))
    jl=np.reshape(jl,(-1,1))
    ix = (il*5 - 8)
    jx = (jl*5 - 8)
    cx = len(lb)
    cy = len(ix)
    lb = matlib.repmat(lb,[1, cy])
    lb=np.reshape(lb,(-1,1))
    kb = matlib.repmat(kb,[1, cy])
    kb=np.reshape(kb,(-1,1))
    ix = matlib.repmat(ix,[cx ,1])
    ix=np.reshape(ix,(-1,1))
    jx = matlib.repmat(jx[cx, 1])
    jx=np.reshape(jx,(-1,1))

    l = lb+jx
    k = kb + ix
    sample_x = round(X + (-l *S *si + k *S *co))
    sample_y = round(Y + (l *S *si + k *S *si))

    xs = round(X + (-(jx+ 1)*S *si +(ix+1)*S*co))
    ys = round(Y +((jx+1)*S *co +(ix+1)*S*si))

    gauss_s1 = gaussian(xs - sample_x, ys - sample_y,2.5*S)
    rx = HaarX(sample_y,sample_x, 2*S,img)
    ry = HaarY(sample_y,sample_x, 2*S,img)

    rrx = np.multiply(gauss_s1,(-rx*si + ry * co))
    rrx = np.reshape(rrx,(cx,cy))
    rry = np.multiply(gauss_s1,(rx*co +ry*si))
    rry = np.reshape(rry,(cx,cy))
    cx = -0.5 + il + 1
    cy = -0.5 + jl + 1
    gauss_s2 = gaussian(cx-2,cy-2,1.5)

    dx = sum(rrx,1)
    dy = sum(rry,1)
    mdx = sum(abs(rrx),1)
    mdy = sum(abs(rry),1)
    dx_yn = 0
    mdx_yn  = 0
    dy_xm = 0
    mdy_xn = 0
    array = np.array([dx, dy,mdx,mdy])
    array = np.reshape(array, (-1,1))
    descriptor = np.multiply(array,np.repmat(gauss_s2,[4,2]))
    lenVal = sum(np.multiply(np.power(dx,2)+np.power(dy,2)+np.power(dx_yn,2)+dy_xm+mdx_yn+mdy_xn),np.power(gauss_s2,2))
    descriptor = np.reshape(descriptor,(-1,1))/np.sqrt(lenVal)
    return descriptor




def gaussian(x,y,sig):
    return np.multiply(1/(2*pi*sig**2), math.exp(np.power(-(np.power(x,2) + np.power(y,2)/(2*sig**2)))))








