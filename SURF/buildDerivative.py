from getResponse import getResponse

def buildDerivative(r,c,t,m,b):
    dx = (getResponse(m,r,c+1,t) - getResponse(m,r,c-1,t))/2
    dy = (getResponse(m,r+1,c,t)- getResponse(m,r-1,c,t))/2
    ds = (getResponse(t,r,c,0) - getResponse(b,r,c,t))/2
    D = [dx, dy, ds]
    return D
