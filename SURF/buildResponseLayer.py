import numpy as np 
def buildResponseLayer(rl, Data):
    from boxIntegral import boxIntegral
    step = np.fix(rl['step'])
    b = np.fix((rl['filter'] - 1)/2 + 1)
    l = np.fix(rl['filter']/3)
    w = np.fix(rl['filter'])
    inverse = 1/np.double(w *w)
    img = Data['img']

    nx,ny = (rl['height'],rl['width'])
    [ar,ac] = np.mgrid[0:nx,0:ny]
    ar = ar.flatten()
    ac = ac.flatten()
    
    r = ar * step
    c = ac * step
    Dxx = boxIntegral(r-l+1,c-b, 2 *l-1,w,img) - boxIntegral(r-l+1, c-np.fix(l/2),2*l-1,l,img) * 3 
    Dyy = boxIntegral(r-b,c-l+1,w,2*l-1,img)-boxIntegral(r-np.fix(l/2),c-l+1,l,2*l-1,img) * 3
    Dxy = boxIntegral(r-1,c+1,l,l,img)+boxIntegral(r+1,c-l,l,l,img)- boxIntegral(r-l,c-l,l,l,img) - boxIntegral(r+1,c+1,l,l,img)
    Dxx = np.multiply(Dxx,inverse)
    Dyy = Dyy *inverse
    Dxy = Dxy *inverse

    rl["responses"] = np.multiply(Dxx, Dyy)- 0.81 * np.multiply(Dxx, Dyy)
    rl["laplacian"] = (Dxx + Dyy) >= 0

    return rl
