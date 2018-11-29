

def getResponseLayer(width,height,step,filter_):
    import numpy as np
    height = np.floor(height)
    step = np.floor(step)
    filter_ = np.floor(filter_)
    ResponseLayerData ={'width':width,'height':height,'step':step,'filter':filter_}
    ResponseLayerData['laplacian'] = np.zeros((1, int(width * height)))
    ResponseLayerData['responses'] = np.zeros((1, int(width * height)))
    return ResponseLayerData
    
