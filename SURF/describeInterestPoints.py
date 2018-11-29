from getDescriptor import getDescriptor
def describeInterestPoints(ipts, upright,extended, img, verbose):
    for i in range (0,len(ipts)):
        ip = ipts[i]
        if(extended):
            ip["descriptorLength"] = 128
        else :
            ip["descriptorLength"] = 64
        ip.descriptor = getDescriptor(ip,upright,extended,img,verbose)
        ipts[i]["orientation"] = ip.orientation
        ipts[i]["descriptor"] = ip.descriptor
    return ipts