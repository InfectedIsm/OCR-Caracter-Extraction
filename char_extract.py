# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 22:13:41 2019

@author: infected
"""
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

debug = False

#xor function to detect transition between 0>>1 and 1>>0 to extract char
def xor(a,b):
    return (not(a) and b) or (a and not(b))

## apply blur filter in order to enbigger character thickness
## then apply the
def clean_image(img) :
    cleaned_img = cv2.blur(img,(3,1))
    cleaned_img = cv2.blur(cleaned_img,(3,3))
    cleaned_img = cv2.adaptiveThreshold(cleaned_img ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,5)
    return cleaned_img

img = cv2.imread("text_lyons.png")
img = img[0:45,:]
# img = img[45:90,:]


# convert image to grayscale in order to treat it easily
# as color are not useful for text
gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#Improve contrast and widen characters
gray_img = clean_image(gray_img)

# Sum pixels on the x axis in order to detect blank zones
x_sum_img = np.sum(gray_img,axis=0)/255

# determine a threshold value for blank zone as some zones are whitter, I move a bit the threshold
blank_threshold = np.max(x_sum_img)*0.935

#Zones where a char is present have a value of 1, blank zone a value of 0

x_separation = (x_sum_img < blank_threshold).astype(int)

#extract index where char appears
#we now there's a char if separation = 1, then we look for index where we jump from 0 to 1 then 1 to 0
prev_value = 0
index_list = []
for index,value in enumerate(x_separation) :
    if xor(value,prev_value):
        index_list = np.concatenate((index_list,[index]))
    prev_value = value

#list that will contain all extracted char
extracted_char = []
number_of_expected_char = int(len(index_list)/2)

#each couple of value [0,1], [2,3], ... from index_list correspond to character
for index in range(number_of_expected_char):
    left_bound = int(index_list[index*2])
    right_bound = int(index_list[index*2+1])
    #Expand the boudaries a little bit on the left and right
    extracted_char.append(gray_img[ : , left_bound -2 :right_bound +2])




#Trace the image + summed pixel values + separation
fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.plot(x_sum_img)
plt.imshow(gray_img,cmap = plt.get_cmap('gray'))
ax.plot(x_separation)

#trace all the char in a plot
columns = 5
rows = int( math.ceil( np.round( len(extracted_char)/5)))
fig = plt.figure(1,figsize=(columns,rows))
for i in range(len(extracted_char)):
    ax = fig.add_subplot(columns,rows,i+1)
    plt.imshow(extracted_char[i], cmap = plt.get_cmap('gray'))
plt.show()