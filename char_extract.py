# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 22:13:41 2019

@author: infected
"""

import numpy as np
import cv2
from scipy import ndimage
from matplotlib import image
import matplotlib.pyplot as plt

## apply blur filter in order to enbigger character thickness
## then apply the
def clean_image(img) :
    cleaned_img = cv2.blur(img,(3,1))
    cleaned_img = cv2.blur(cleaned_img,(3,3))
    cleaned_img = cv2.adaptiveThreshold(cleaned_img ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,5)
    return cleaned_img

img = cv2.imread("text_lyons.png")
# img = img[0:35,:]
img = img[45:90,:]


# convert image to grayscale in order to treat it easily
# as color are not useful for text
gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
print(np.shape(gray_img))

# equalize histogram for higher contrast
# gray_img = cv2.equalizeHist(gray_img)
gray_img = clean_image(gray_img)


max = np.max(gray_img,0)
indice = np.argmax(gray_img,0)
x_max_img = np.concatenate((max,indice))

# Sum pixels on the x axis in order to detect blank zones
x_sum_img = np.sum(gray_img,axis=0)/255
print(gray_img)
# trace sumed vector over the image
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(x_sum_img)
plt.imshow(gray_img,cmap = plt.get_cmap('gray'))

# determine a threshold value for blank zone
# as some zones are whitter, I move a bit the threshold
blank_threshold = np.max(x_sum_img)*0.935
print(blank_threshold)
separation = (x_sum_img < blank_threshold).astype(int)*blank_threshold
ax.plot(separation)
plt.show()

