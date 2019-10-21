# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 22:13:41 2019

@author: infected
"""
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

##
def xor(a,b):
    """
    perform xor operation to detect transition between 0>>1 and 1>>0 to extract char

    :param a: (int)
    :param b: (int)
    :return: result – (bool)
    """
    return (not(a) and b) or (a and not(b))

def SaveCharToPNG(img, path):
    img = cv2.resize(img, (140, 140))
    path = str(path)
    cv2.imwrite(path, img)

def clean_image(img) :
    """
    Apply blur filter in order to enlarge character thickness, then threshold the image into
    a two value (0 & 255) image.

    :param img: (array)
    :return: (array)
    """
    cleaned_img = cv2.blur(img,(3,1))
    cleaned_img = cv2.blur(cleaned_img,(3,3))
    cleaned_img = cv2.adaptiveThreshold(cleaned_img ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,5)
    return cleaned_img


def detect_changes(separation):
    """

    Take as input a boolean vector that represent zones where char have been detected (1)
    And zone where nothing have been detected (0)
    Then it extracts indexes where a a zone is crossed (0 -> 1 or 1 -> 0)
    As 0 means blank, and 1 means char, 0->1->0 means we found a char
    This function is also used to find lines in a document

    :param separation: (array) threshold vector
    :return: (array) indexes of sub-img limits
    """

    prev_value = 0
    index_list = []
    for index,value in enumerate(separation) :
        if xor(value,prev_value):
            index_list = np.concatenate((index_list,[index]))
        prev_value = value
    return index_list

def separate(img, threshold, axis=0):
    """
    Separate a text image into sub text-images

    :param img: (array) image to separate into parts
    :param threshold: (float) threshold for separation, from 0 to 1
    :param axis: (int) 0 = x axis, 1 = y axis
    :return: index of separation
    """

    # Sum pixels on the y axis in order to detect blank zones (space between lines)
    x_sum_img = np.sum(img, axis=axis) / 255

    # determine a threshold value for blank zone
    # (as black=0, white=255, blank zone will have a much higher summed value)
    x_blank_threshold = np.max(x_sum_img) * threshold

    # Zones where a char is present have a value of 1, blank zone a value of 0
    x_separation = (x_sum_img < x_blank_threshold).astype(int)

    # extract index where char appears
    # we now there's a char if separation = 1, then we look for index where we jump from 0 to 1 then 1 to 0
    separation_index_list = detect_changes(x_separation)

    return separation_index_list


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


def extract_doc2lines(gray_img, is_cleaned=True, raw_img=np.array([])):
    """
    Take as input a document (image or pdf) and give back all the extracted lines as images

    :param gray_img: (array) image you want to treat
    :param is_cleaned: (array) if true, the image will not be cleaned by the function
    :param raw_img: (array) necessary if "is cleaned" is True
    :return:
    """

    #I want to allow the user to be able clean the image by himself
    #If he chose to clean itself (False),

    if is_cleaned == False:
        raw_img = gray_img
        gray_img = clean_image(gray_img)

    #if the user forget to put a raw_img, replace raw_img with gray_img
    # It is important to keep the raw image as the "cleaning" add blur to the char
    if (is_cleaned == True ) & ( np.shape(raw_img)==0 ):
        raw_img = gray_img

    y_index_list = separate(gray_img, threshold=0.935, axis=1)

    number_of_expected_line = int(len(y_index_list)/2)

    # This loop extract lines thanks to idexes listed in index_list
    # Each couple of index represent char zone left and right boundaries
    extracted_line = []
    for index in range(number_of_expected_line):
        upper_bound = int(y_index_list[index*2])
        lower_bound = int(y_index_list[index*2+1])
        #Expand the boudaries a little bit on the left and right
        extracted_line.append(raw_img[upper_bound -2 :lower_bound +2, :])

    return extracted_line

def extract_lines2words(line_img):
    """"
    The following function take as input a line (image) and give back all the extracted words as images
    input > lines : 2D array or list of 2D arrays
    output > extracted_words : list of 2D arrays
    """

    # Now that we have extracted lines, we can apply the same algorithm as above, but for each line
    # hence the additional for loop running through extracted_line
    extracted_word = []

    line_raw_img = line_img
    line_img = clean_image(line_img)

    # bluring the image horizontally in order to merge char into words, allowing then to detect words
    line_img_blurred = cv2.blur(line_img, (4, 1))
    line_img_blurred = cv2.blur(line_img_blurred, (4, 1))

    word_index_list = separate(line_img_blurred, threshold=0.935, axis=0)

    number_of_expected_word = int(len(word_index_list) / 2)

    # create a key for each line
    for index in range(number_of_expected_word):
        left_bound = int(word_index_list[index*2])
        right_bound = int(word_index_list[index*2+1])
        #Here I don't use the blured image, as it will become impossible to
        #extract chars, that's why I extract words from line_img and not line_img_blured
        extracted_word.append(line_raw_img[ : , left_bound -2 :right_bound +2])

    return extracted_word

def extract_words2char(word_img):
    extracted_char = []

    word_raw_img = word_img
    word_img_cleaned = clean_image(word_img)

    x_index_list= separate(word_img_cleaned,threshold=0.935, axis=0)

    number_of_expected_char = int(len(x_index_list)/2)

    #each couple of value [0,1], [2,3], ... from index_list correspond to character
    for index in range(number_of_expected_char):
        left_bound = int(x_index_list[index*2])
        right_bound = int(x_index_list[index*2+1])
        char_img = word_raw_img[ : , left_bound -1 :right_bound +1]
        char_img = cv2.copyMakeBorder(char_img,1,1,1,1, borderType=cv2.BORDER_CONSTANT
                                                        ,value=[255,255,255])
        #Expand the boudaries a little bit on the left and right
        extracted_char.append(char_img)

    return extracted_char


# dictionary that will contain an ordered list of char, sorted by words and lines
# document = {}
# n_lines=0
# n_words=0
# n_char=0
#
# img = cv2.imread("document_images/text_lyons.png")
# # convert image to grayscale in order to treat it easily
# # as color are not useful for text
# gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# #Improve contrast and widen characters
# gray_img = clean_image(gray_img)
#
# extracted_line = extract_doc2lines(gray_img)
#
# for line_idx,line_img in enumerate(extracted_line):
#     extracted_word = extract_lines2words(line_img)
#     n_lines+=1
#     document[line_idx] = {}
#
#     for word_idx,word_img in enumerate(extracted_word):
#         n_words += 1
#         extracted_char= extract_words2char(word_img)
#         document[line_idx][word_idx] = {}
#
#         for char_idx, char_img in enumerate(extracted_char):
#             n_char += 1
#             document[line_idx][word_idx][char_idx] = char_img
#
# # print(document)
#
# trace_all_char(extracted_char)
#
# print("lines:", n_lines,
#       "\nwords", n_words,
#       "\nchar:", n_char)
#
