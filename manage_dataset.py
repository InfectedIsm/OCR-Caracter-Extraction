import os
import glob

import numpy as np
import cv2 as cv
import pandas as pd

# Generate dataset file from images

#Root folder for the images
__files_path = 'dataset/English/Fnt/'
# this line count how many files there is in the path
__total_folders = len([name for name in os.listdir(__files_path) if os.path.isdir(os.path.join(__files_path, name))])

#Futures files paths
__dataset_path = 'dataset/' + 'train_datas' + '.csv'
__labels_path = 'dataset/' + 'labels' + '.csv'
__table_path = 'dataset/' + 'table' + '.csv'

#This function generate a randomly shuffled array of indexes with no repetition
def GenerateRandomIndexes(size,max):
    return np.random.choice(np.arange(max), replace=False, size=size)

#This function read PNG images and convert them to CSV files where each image
#is encoded on a vector
def CreateCsvFromImg(chunk_size=100, debug=False):

    for dir in range(1, __total_folders +1):
        dir_path = 'dataset/English/Fnt/Sample' + str(f"{dir:03d}") +'/'
        csv_path = __files_path +'char' + str(f"{dir:03d}") + '.csv'
        # this line count how many png there is in the path
        if os.path.isfile(csv_path):
            os.remove(csv_path)
        # this line count how many png there is in the path
        total_images = len(glob.glob1(dir_path,'*.png'))
        print('Sample' + str(f"{dir:03d}") + ' Total files=' + str(total_images))

        for i in range(1,total_images+1,chunk_size):
            # create empty array to store data temporarily
            datas = np.zeros((0, 784))

            for j in range(chunk_size):
                path = dir_path + 'img' + str(f"{dir:03d}") + '-' + str(f"{i+j:05d}") + '.png'
                img = cv.imread(path)
                img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                img = cv.resize(img, (28, 28))
                img_vector = np.reshape(img, (1,784))
                datas = np.append(datas,img_vector,axis=0)
                df = pd.DataFrame(datas)
                # Stop the loop when no more files to read
                if (i + j) == total_images:
                    break

                # mode=a >> append data to csv file

            df.to_csv(csv_path, index=False, header=False, mode='a', chunksize=chunk_size)
        print(np.shape(df.values))
        file_size = os.path.getsize(csv_path)
        print(str(file_size >> 10) + ' kb')  # small trick to convert bytes to kb

    return 0

def CreateDataset(training_samp_by_char=100,random=True, debug=False):

    #remove train file if exists
    if os.path.isfile(__dataset_path):
        os.remove(__dataset_path)
    #create new train file
    with open(__dataset_path,'w'):
        pass

    # create new labels file
    if os.path.isfile(__labels_path):
        os.remove(__labels_path)
    with open(__labels_path,'w'):
        pass

    #Here I use __total_folder as there's as many csv as folders
    for file in range(1,__total_folders+1):
        csv_path = __files_path + 'char' + str(f"{file:03d}") + '.csv'
        # if a csv file is missing, abort function and delete train set
        if not(os.path.isfile(csv_path)):
            return -1

        datas = np.zeros((0, 784))

        if random:
            #generate a vector of "size" sample, picked between 0 and 1015
            indexes_to_remove = np.random.choice(np.arange(0, 1016), replace=False, size=training_samp_by_char+1)
            # print("rem:"+str(np.shape(indexes_to_remove)[0]))
            #generate a vector of 1016 element from 0 to 1015
            all_indexes = np.arange(0,1016)
            # print("all:"+str(np.shape(all_indexes)[0]))
            #remove element at index "indexes_to_remove"
            indexes_to_skip = np.delete(all_indexes,indexes_to_remove)
            # print("skip:"+str(np.shape(indexes_to_skip)[0]))
            df1 = pd.read_csv(csv_path, skiprows=indexes_to_skip)

        else:
            indexes_to_remove = np.arange(training_samp_by_char)

        #In the same time as the dataset, a label file is generated where each lines
        #corespond to each others
        labels = np.ones(training_samp_by_char,dtype=int)*file
        df2 = pd.DataFrame(labels)

        df1.to_csv(__dataset_path, index=False, header=False, mode='a', chunksize=training_samp_by_char)
        # print("write:"+str(np.shape(df1.values)[0]))
        df2.to_csv(__labels_path, index=False, header=False, mode='a', chunksize=training_samp_by_char)
        print(str(file) + ' char treated')
    return 0

# Read a CSV file and return a np.ndarray
def CsvToArray(dataset_csv=__dataset_path, labels_csv=__labels_path, percentage=1):
    df = pd.read_csv(dataset_csv)
    data = df.values
    df = pd.read_csv(labels_csv)
    labels = df.values

    return data[0:int(np.shape(data)[0]*percentage)-1,:], labels[0:int(np.shape(labels)[0]*percentage)-1,:]

def IndexToChar():
    df = pd.read_csv(__table_path, header=None)
    char_index = df.values

    return char_index
