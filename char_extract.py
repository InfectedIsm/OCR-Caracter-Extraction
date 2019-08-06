# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 22:13:41 2019

@author: infected
"""
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

##xor function to detect transition between 0>>1 and 1>>0 to extract char
def xor(a,b):
    return (not(a) and b) or (a and not(b))

## apply blur filter in order to enbigger character thickness
## then apply the
def clean_image(img) :
    cleaned_img = cv2.blur(img,(3,1))
    cleaned_img = cv2.blur(cleaned_img,(3,3))
    cleaned_img = cv2.adaptiveThreshold(cleaned_img ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,5)
    return cleaned_img

## This function take as input a boolean vector that represent zones where char have been detected (1)
## And zone where nothing have been detected (0)
## Then it extracts indexes where a a zone is crossed (0 -> 1 or 1 -> 0)
## As 0 means blank, and 1 means char, 0->1->0 means we found a char
## This function is also used to find lines in a document
def detect_changes(separation):
    prev_value = 0
    index_list = []
    for index,value in enumerate(separation) :
        if xor(value,prev_value):
            index_list = np.concatenate((index_list,[index]))
        prev_value = value
    return index_list

def trace_all_char(char_list, num_col = 10) :
    # trace all the char in a plot
    num_col = 10
    rows = int( math.ceil( np.round( len(char_list)/5)))
    fig = plt.figure(1,figsize=(num_col,rows))
    for i in range(len(char_list)):
        ax = fig.add_subplot(num_col,rows,i+1)
        plt.imshow(char_list[i], cmap = plt.get_cmap('gray'))

    # Trace the image + summed pixel values + x_separation
    # fig = plt.figure(0)
    # ax = fig.add_subplot(111)
    # ax.plot(x_sum_img)
    # plt.imshow(line_img, cmap=plt.get_cmap('gray'))
    # ax.plot((x_separation * 25))
    plt.show()

def extract_doc2lines(gray_img):
    # Sum pixels on the y axis in order to detect blank zones (space between lines)
    y_sum_img = np.sum(gray_img,axis=1)/255

    # determine a threshold value for blank zone
    # (as black=0, white=255, blank zone will have a much higher summed value)
    y_blank_threshold = np.max(y_sum_img)*1

    # Zones where a char is present have a value of 1, blank zone a value of 0
    y_separation = (y_sum_img < y_blank_threshold).astype(int)

    # extract index where char appears
    # we now there's a char if separation = 1, then we look for index where we jump from 0 to 1 then 1 to 0
    y_index_list = detect_changes(y_separation)
    number_of_expected_line = int(len(y_index_list)/2)

    # This loop extract lines thanks to idexes listed in index_list
    # Each couple of index represent char zone boundaries
    extracted_line = []
    for index in range(number_of_expected_line):
        upper_bound = int(y_index_list[index*2])
        lower_bound = int(y_index_list[index*2+1])
        #Expand the boudaries a little bit on the left and right
        extracted_line.append(gray_img[upper_bound -2 :lower_bound +2, :])

    return extracted_line

def extract_lines2words(lines):
    # Now that we have extracted lines, we can apply the same algorithm as above, but for each line
    # hence the additional for loop running through extracted_line
    extracted_word = []
    #bluring the image horizontally in order to merge char, allowing then to detect words
    line_img_blurred = cv2.blur(line_img, (4, 1))
    line_img_blurred = cv2.blur(line_img_blurred, (4, 1))
    x_sum_img = np.sum(line_img_blurred, axis=0) / 255
    x_blank_threshold = np.max(x_sum_img) * 0.935
    x_separation = (x_sum_img < x_blank_threshold).astype(int)
    word_index_list = detect_changes(x_separation)

    #create a key for each line

    number_of_expected_word = int(len(word_index_list) / 2)

    for index in range(number_of_expected_word):
        left_bound = int(word_index_list[index*2])
        right_bound = int(word_index_list[index*2+1])
        #Here I don't use the blured image, as it will become impossible to
        #extract chars, that's why I extract words from line_img and not line_img_blured
        extracted_word.append(line_img[ : , left_bound -2 :right_bound +2])

    return extracted_word

def extract_words2char(word_img):
    extracted_char = []

    x_sum_img = np.sum(word_img,axis=0)/255
    x_blank_threshold = np.max(x_sum_img)*0.935
    x_separation = (x_sum_img < x_blank_threshold).astype(int)
    x_index_list = detect_changes(x_separation)

    number_of_expected_char = int(len(x_index_list)/2)

    #each couple of value [0,1], [2,3], ... from index_list correspond to character
    for index in range(number_of_expected_char):
        left_bound = int(x_index_list[index*2])
        right_bound = int(x_index_list[index*2+1])
        #Expand the boudaries a little bit on the left and right
        extracted_char.append(word_img[ : , left_bound -2 :right_bound +2])

    return extracted_char


#dictionary that will contain an ordered list of char, sorted by words and lines
document = {}
n_lines=0
n_words=0
n_char=0

img = cv2.imread("document_images/bigger_text.png")
# convert image to grayscale in order to treat it easily
# as color are not useful for text
gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#Improve contrast and widen characters
gray_img = clean_image(gray_img)

extracted_line = extract_doc2lines(gray_img)

for line_idx,line_img in enumerate(extracted_line):
    extracted_word = extract_lines2words(line_img)
    n_lines+=1
    document[line_idx] = {}

    for word_idx,word_img in enumerate(extracted_word):
        n_words += 1
        extracted_char= extract_words2char(word_img)
        document[line_idx][word_idx] = {}

        for char_idx, char_img in enumerate(extracted_char):
            n_char += 1
            document[line_idx][word_idx][char_idx] = char_img

# print(document)

# trace_all_char(extracted_char,10)

print("lines:", n_lines,
      "\nwords", n_words,
      "\nchar:", n_char)

