import numpy as np

def getResponse(a, row, column, b):
    if (b == 0):
        scale = 1
    else:
        scale = np.fix(int(a['width'])/int(b['width']))
    index = np.fix(scale*row)*a['width'] + np.fix(scale*column)+1
    index = np.asarray(index,dtype=int)
    an = a['responses'][index]
    return an
