# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 22:13:41 2019

@author: infected
"""

import cv2

from neural_network import UseModel
from char_extract import *

document = {}
n_lines=0
n_words=0
n_char=0

img = cv2.imread("document_images/text_lyons.png")
# convert image to grayscale in order to treat it easily
# as color are not useful for text
text_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#Improve contrast and widen characters

extracted_line = extract_doc2lines(text_img, is_cleaned=False, raw_img=text_img)

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
            path = 'result/char_l{}w{}c{}.png'.format(line_idx, word_idx, char_idx)
            # print(np.shape(char_img))
            char_img = cv2.resize(char_img,(28, 28))
            document[line_idx][word_idx][char_idx] = char_img
            SaveCharToPNG(char_img, path)

            # print(np.shape(char_img))

# print(document)

print("lines:", n_lines,
      "\nwords", n_words,
      "\nchar:", n_char)


with open("result/text.txt","w") as f :
    for line_number, line in document.items():
        for word_number, word in line.items():
            word_buffer = ""
            for char_number, char in word.items():
                word_buffer = word_buffer + UseModel(char)
            print(word_buffer, end=' ', file=f)
            # print(word_buffer)
        print(file=f)
        # print()

