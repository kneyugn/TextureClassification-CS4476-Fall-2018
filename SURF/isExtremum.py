import numpy as np 


def isExtremum(r,c,t,m,b,Data):
    layerBorder = np.fix((t['filter'] + 1)/ (2 * t['step']))
    bound_check_fail = np.logical_or(np.logical_or(r <= layerBorder,r >= t['height'] - layerBorder),np.logical_or(c <= layerBorder, c >=t["width"] - layerBorder))
    candidate = getResponse(m,r,c,t)
    threshold_fail = candidate < Data['thresh']
    an = np.logical_and(np.logical_not( bound_check_fail),np.logical_not(threshold_fail))
    for rr in range (-1,2):
        for cc in range(-1,2):
            check1 = getResponse(t, r + rr, c + cc, t) >= candidate
            check2 = getResponse(m, r + rr, c + cc, t) >= candidate
            check3 = getResponse(b, r + rr, c + cc, t) >= candidate
            check4 = np.logical_or(rr != 0,cc !=0)
            an3 = np.logical_not(np.logical_or(np.logical_or(check1,np.logical_and(check4,check2)),check3))
            an = np.logical_and(an,an3)
    print "aaannn", np.shape(an)
    return an


def getResponse(a, row, column,b):
    
    scale = np.fix(a['width']/b['width'])
    index = np.fix(scale*row)*a['width'] + np.fix(scale*column)+1
    index[index < 0] = 0
    index[index >= len(a['responses'])] = len(a['responses'])- 1
    index = np.asarray(index,dtype=int)
    an = a['responses'][index]
    return an
