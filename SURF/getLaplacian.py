
import numpy as np

def getLaplacian(a,row,column,b):
    if b == 0:
        scale = 1
    else:
        scale = np.fix(a["width"]/b["width"])

    an = a["laplacian"][int(np.fix(scale*row)* a["width"] +np.fix(scale * column)+1)]
    return an
