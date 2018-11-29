from boxIntegral import boxIntegral
def HaarX(row, column,size,img):
    return boxIntegral(row - size /2, column,size,size/2,img) - boxIntegral(row - size / 2, column - size / 2, size, size / 2, img)