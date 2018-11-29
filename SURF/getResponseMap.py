import numpy as np 

def getResponseMap(Data):
    from getResponseLayer import getResponseLayer
    from buildResponseLayer import buildResponseLayer
    responseMap = []

    width = (np.shape(Data['img'])[1])/Data['init_sample']
    height = (np.shape(Data['img'])[0])/Data['init_sample']
    s = Data['init_sample']

    

    if Data['octaves'] >= 1:
        responseMap.append(getResponseLayer(width, height, s, 9))
        responseMap.append(getResponseLayer(width, height, s, 15))
        responseMap.append(getResponseLayer(width, height, s, 21))
        responseMap.append(getResponseLayer(width, height, s, 27))
    if Data['octaves'] >= 2:
        responseMap.append(getResponseLayer(width/2, height/2, s * 2, 39))
        responseMap.append(getResponseLayer(width/2, height/2, s * 2, 51))
    if Data['octaves'] >= 3:
        responseMap.append(getResponseLayer(width/4, height/4, s * 2, 75))
        responseMap.append(getResponseLayer(width/2, height/4, s * 4, 99))
    if Data['octaves'] >= 4:
        responseMap.append(getResponseLayer(width/8, height/8, s * 8, 147))
        responseMap.append(getResponseLayer(width/8, height/8, s * 8, 195))
    if Data['octaves'] >= 5:
        responseMap.append(getResponseLayer(width/16, height/16, s * 16, 291))
        responseMap.append(getResponseLayer(width/16, height/16, s * 16, 387))

    for i in range(len(responseMap)):
        responseMap[i] = buildResponseLayer(responseMap[i], Data)

    return responseMap 


     
        
        
    
        

