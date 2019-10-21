# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 22:13:41 2019

@author: infected
"""

############# INITIALISATION #############
#=========================================

## System library imports
import os
import glob

## Numpy, OpenCV and Pandas import
import numpy as np
import cv2 as cv
import pandas as pd
#=========================================

## Local Variables

# Root folder for the images
__files_path = 'dataset/English/Fnt'
__wrong_char_folder = 'dataset/English/Fnt/Sample{}/WrongCharacters'
# this line count how many directories there is in the path
__total_folders = len([name for name in os.listdir(__files_path)
                       if os.path.isdir(os.path.join(__files_path, name))])

#Futures files paths
__dataset_path = 'dataset/train_datas.csv'
__labels_path = 'dataset/labels.csv'
__table_path = 'dataset/table.csv'
__bad_data_path = 'dataset/bad_data_to_delete.csv'
#=========================================



############# MODULE DEFINITIONS #############
#=============================================

def CreateOrWriteOverFile(path):
    """If the file exists, delete it and write it. Else, just write the file

    :param path: path for the file to create
    :return:
    """
    if os.path.isfile(path):
        os.remove(path)
    # create new train file
    with open(path,'w'):
        pass

    return 0


# This function generate a randomly shuffled array of indexes with no repetition
def GenerateRandomIndexes(size,max):
    return np.random.choice(np.arange(max), replace=False, size=size)

def CleanDatasetFromWrongChar():
    """Clean dataset from bad characters

    Read a file containing indexes of wrong characters that I have
    detected in the dataset, and remove them.
    For example, for some fonts, lowercase char are exactly the same as uppercase ones
    There's also very strange fonts that are unreadable
    I remove all those chars, it has been chosen manually
    """

    # Read all bad datas indexes stored in csv file (.read_csv) and convert the resulting dataframe in an array (.value)
    bad_datas_indexes = pd.read_csv('dataset/bad_data_to_delete.csv').values

    # Run through all dataset folders
    for dir in range(1, __total_folders + 1):
        # dir_path = 'dataset/English/Fnt/Sample' + str(f"{dir:03d}") + '/'
        dir_path = "{}/Sample{:03d}".format(__files_path, dir)
        dir_path = os.path.abspath(dir_path)
        # This list contains all img paths in sorted order
        png_list = sorted(glob.glob(dir_path+"/*png"))
        png_moved = 0

        for index, file in enumerate(png_list):
            # read the 5 number at the end of the png file and check if considered as bad data
            if int(file[-9:-4]) in bad_datas_indexes - 1:

                parent_folder, file_name = os.path.split(os.path.abspath(file))
                wrong_char_folder = "{}/WrongCharacters/".format(parent_folder)

                # Create target Directory if don't exist
                if not os.path.exists(wrong_char_folder):
                    os.mkdir(wrong_char_folder)

                os.rename(os.path.abspath(file),"{}/{}".format(wrong_char_folder, file_name))
                png_moved+=1

        print("{} files moved for {}".format(png_moved,os.path.split(dir_path)[1]))

    return 0

def RestoreOriginalDataset():
    """Restore modifications on the dataset done by CleanDatasetFromWrongChar()

    """
    for dir in range(1, __total_folders + 1):
        dir_path = "{}/Sample{:03d}".format(__files_path, dir)
        dir_path = os.path.abspath(dir_path)
        wrong_char_folder = "{}/WrongCharacters/".format(dir_path)
        # This list contains all img paths in sorted order
        png_list = sorted(glob.glob(wrong_char_folder+"*png"))
        png_moved = 0

        for index, file in enumerate(png_list):
                parent_folder, file_name = os.path.split(os.path.abspath(file))
                samples_folder = os.path.split(parent_folder)[0]

                # move files from WrongCharacters folder to parent folder
                os.rename(os.path.abspath(file),"{}/{}".format(samples_folder, file_name))
                png_moved+=1

        print("{} files restored for {}".format(png_moved,os.path.split(dir_path)[1]))

    return 0

def CreateCsvFromPNG(chunk_size=100):
    """create a CSV file for each existing char dataset

    Where each line is a vectorized image

    :param chunk_size: batch number of image read and then wrote in the csv file
    :return: nothing
    """
    for dir in range(1, __total_folders +1):
        dir_path = "{}/Sample{:03d}".format(__files_path, dir)
        csv_path = "{}/char{:03d}.csv".format(__files_path, dir)

        # this line count how many png there is in the path
        if os.path.isfile(csv_path):
            print("file already existing. File is deleted to create a new one.")
            os.remove(csv_path)
        with open(csv_path, 'w'):
            pass
        # this line count how many png there is in the path
        images_names = glob.glob1(dir_path,'*.png')
        total_images = len(images_names)
        print('Sample' + str(f"{dir:03d}") + ' Total files=' + str(total_images))

        df = pd.DataFrame()
        for i in range(0,total_images,chunk_size):
            # create empty array to store data temporarily
            datas = np.zeros((0, 784))

            for img_name in images_names[i:i+chunk_size]:
                # path = dir_path + '/img' + str(f"{dir:03d}") + '-' + str(f"{i+j:05d}") + '.png'
                path = "{}/{}".format(dir_path, img_name)
                # path=img_path
                # print(os.path.abspath(path))
                img = cv.imread(path)
                img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                img = cv.resize(img, (28, 28))
                img_vector = np.reshape(img, (1,784))
                datas = np.append(datas,img_vector,axis=0)
                df = pd.DataFrame(datas)
                # # Stop the loop when no more files to read
                # if (i + j) == total_images:
                #     break

                # mode=a >> append data to csv file

            df.to_csv(csv_path, index=False, header=False, mode='a', chunksize=chunk_size)
        file_size = os.path.getsize(csv_path)
        print(str(file_size >> 10) + ' kB')  # small trick to convert bytes to kB

    return 0

def CreateDataset(training_samp_by_char=100,random=True, debug=False):
    """
    Read all CSV files generated by CreateCsvFromPNG() and create a new unique CSV file containing the dataset

    :param training_samp_by_char:   (int) Number of sample to put in the dataset for each character
    :param random:                  If True, chose the chars randomly. If False, take the first ones.
    :param debug:                   Debug purpose
    :return:                        nothing
    """
    # remove train file if exists
    if os.path.isfile(__dataset_path):
        print("Dataset file already existing. File is deleted.")
        os.remove(__dataset_path)
    # create new train file
    with open(__dataset_path,'w'):
        print("New dataset file created.")
        pass

    # create new labels file
    if os.path.isfile(__labels_path):
        print("Labels file already existing. File is deleted..")
        os.remove(__labels_path)
    with open(__labels_path,'w'):
        print("New labels file created.")
        pass

    # Here I use __total_folder as there's as many csv as folders
    for dir in range(1,__total_folders+1):
        dir_path = "{}/Sample{:03d}".format(__files_path, dir)
        total_images = len(glob.glob1(dir_path, '*.png'))

        #error handling for wrong values
        if training_samp_by_char==0 or training_samp_by_char > total_images:
            print(
                "Warning: as you asked for {} samples, but there's {}, {} has been chosen"
                    .format(training_samp_by_char, total_images, total_images))

            training_samp_by_char = total_images

        csv_path = "{}/char{:03d}.csv".format(__files_path, dir)
        # if a csv file is missing, abort function and delete train set
        if not(os.path.isfile(csv_path)):
            return -1

        df1= pd.DataFrame()
        if random:
            # generate a vector of "size" sample, picked between 0 and 1015
            indexes_to_remove = np.random.choice(np.arange(0, total_images),
                                                 replace=False,
                                                 size=training_samp_by_char)

            # generate a vector of 1016 element from 0 to 1015
            all_indexes = np.arange(0, total_images)

            # remove element at index "indexes_to_remove"
            indexes_to_skip = np.delete(all_indexes,indexes_to_remove)


            df1 = pd.read_csv(csv_path, skiprows=indexes_to_skip, header=None)

        else:
            df1 = pd.read_csv(csv_path, nrows=training_samp_by_char, header=None)

        # In the same time as the dataset, a label file is generated where each lines
        # corresponds to each others
        labels = np.ones(training_samp_by_char,dtype=int)*dir
        df2 = pd.DataFrame(labels)

        df1.to_csv(__dataset_path, index=False, header=False, mode='a', chunksize=training_samp_by_char)
        # print("write:"+str(np.shape(df1.values)[0]))
        df2.to_csv(__labels_path, index=False, header=False, mode='a', chunksize=training_samp_by_char)
        print("Sample{:03d} treated".format(dir))

        print("dataset size: {} samples".format(df1.shape[0]))
    return 0

# Read a CSV file and return a np.ndarray
def CsvToArray(dataset_csv=__dataset_path, labels_csv=__labels_path, percentage=1):
    """
    Convert the dataset CSV to numpy array

    :param dataset_csv: (path)
    :param labels_csv:  (path)
    :param percentage:  (int) from 0 to 1. Percentage of the dataset to take.
    :return:            (np.array) dataset and labels in np.array format
    """
    df = pd.read_csv(dataset_csv)
    data = df.values
    df = pd.read_csv(labels_csv)
    labels = df.values

    return data[0:int(np.shape(data)[0]*percentage)-1,:], labels[0:int(np.shape(labels)[0]*percentage)-1, :]

def IndexToChar():
    """
    Read a CSV file containing the char('0' to 'Z') corresponding to the label (0 to 62)

    :return: char_index – (list of str) list of all possible char where indexes correspond to the label
    """
    df = pd.read_csv(__table_path, header=None)
    char_index = df.values

    return char_index

# CleanDatasetFromWrongChar()
def GenerateAll(skip_PNG_to_CSV=True):
    """ Generate the training data and label CSV files

    :param skip_PNG_to_CSV: skip the png to csv conversion step
    :return: nothing
    """
    if skip_PNG_to_CSV:
        pass
    else:
        CreateCsvFromPNG()

    CreateDataset(training_samp_by_char=0)



GenerateAll()