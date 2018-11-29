import numpy as np
from buildDerivative import buildDerivative
from buildHessian import buildHessian
from getLaplacian import getLaplacian
def interpolateExtremum(r,c,t,m,b,ipts,ap):
    D = buildDerivative(r,c,t,m,b)
    H = buildHessian(r,c,t,m,b)
    Of = np.divide(-H,D)
    O = [Of[0][0], Of[1][0],Of[2][0]]
    filterStep = np.fix((m["filter"] - b["filter"]))
    if (abs(O[0]) < 0.5 and abs(O[1]) < 0.5 and abs(O[2]) < 0.5):
        ipts["x"] = np.double(((c+O[0])) * t["step"])
        ipts["y"] = np.double(((r+O[1])) * t["step"])
        ipts["scale"] = np.double(((2/15) * (m["filter"] +O[2] * filterStep)))
        ipts["laplacian"] = np.fix(getLaplacian(m,r,c,t))

    return ipts, ap
