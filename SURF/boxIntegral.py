import numpy as np 

def boxIntegral(row, col,rows,cols,img):
    import numpy as np 

    row = np.fix(row)
    col = np.fix(col)
    rows = np.fix(rows)
    cols = np.fix(cols)

    rl = np.minimum(row, np.shape(img)[0] - 1)
    cl = np.minimum(row, np.shape(img)[1] - 1)
    r2 = np.minimum(row + rows, np.shape(img)[0] - 1)
    c2 = np.minimum(col +cols, np.shape(img)[1] - 1)


    sx = np.shape(img)[0]

    img = img.flatten()
    maxValue = np.array(np.maximum(rl +(cl-1)*sx,0))
    A = img[np.array(np.maximum(rl +(cl-1)*sx,0),dtype = int)]
    B = img[np.array(np.maximum(rl +(c2-1)*sx,0),dtype = int)]
    C = img[np.array(np.maximum(r2 +(cl-1)*sx,0),dtype = int)]
    D = img[np.array(np.maximum(r2 +(c2-1)*sx,0),dtype = int)]

    A[np.asarray(np.where(rl < 0),dtype = int)] = 0
    A[np.asarray(np.where(cl < 0),dtype = int)] = 0
    B[np.asarray(np.where(rl < 0),dtype = int)] = 0
    B[np.asarray(np.where(c2 < 0),dtype = int)] = 0
    C[np.asarray(np.where(r2 < 0),dtype = int)] = 0
    C[np.asarray(np.where(cl < 0),dtype = int)] = 0
    D[np.asarray(np.where(r2 < 0),dtype = int)] = 0
    D[np.asarray(np.where(c2 < 0),dtype = int)] = 0

    maximum = A-B-C+D

    return maximum




 



    
