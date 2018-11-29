

def getInterestPoints(Data, verbose):
    import numpy as np
    from getResponseMap import getResponseMap
    from isExtremum import isExtremum
    from interpolateExtremum import interpolateExtremum

    print "getInterestPoints"
    filter_map = np.asarray([[0,1,2,3],
     [1,3,4,5],
     [3,5,6,7],
     [5,7,8,9],
     [7,9,10,11]])
    filter_map += 1
    ap = 0
    ipts = {}

    responseMap = getResponseMap(Data)
    for octave in range (0, Data['octaves']):
        for i in range (0,1):
            b = responseMap[filter_map[octave][i]]
            m = responseMap[filter_map[octave][i+1]]
            t = responseMap[filter_map[octave][i + 2]]
            v1 = t["width"] -1
            v2 = t["height"]-1
            [x1,x2] = np.mgrid[0:v1,0:v2]
            x1 = np.reshape(x1, (1,-1))
            x2 = np.reshape(x2, (1,-1))
            p = np.flatnonzero(isExtremum(x1,x2,t,m,b,Data))



            for j in range (len(p)):
                ind= p[j]
                [ipts,ap] = interpolateExtremum(x1[0][ind],x2[0][ind],t,m,b,ipts,ap)
    return ipts
        
            
             
                        

    
