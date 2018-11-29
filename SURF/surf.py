from skimage.color import rgb2gray
import numpy as np
from PIL import Image
from getInterestPoints import getInterestPoints
from describeInterestPoints import describeInterestPoints

def surf(img):
    Data ={'thresh':0.0002,'octaves':5,'init_sample':2,'upright':False, 'extended':False, 'verbose': False}
    integral_image = calculateIntegralImage(img)
    Data['img'] = integral_image
    ipts = getInterestPoints(Data, Data['verbose'])
    ipts = describeInterestPoints(ipts, Data["upright"],Data["extended"],integral_image,Data["verbose"])
    return ipts
def calculateIntegralImage (img):
    return np.array(np.cumsum(np.cumsum(img,0),1))
    
    
img = Image.open("testc1.png")
array = np.asarray(img)
gray = rgb2gray(array)
ipts = surf(gray)
print "pts",ipts
