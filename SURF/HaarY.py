from boxIntegral import boxIntegral

def HaarY(row, column,size, img):
    return boxIntegral(row, column - size / 2, size / 2, size, img) - boxIntegral(row - size / 2, column - size / 2, size / 2, size, img);