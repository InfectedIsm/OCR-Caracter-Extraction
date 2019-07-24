import os
import glob

import numpy as np
import cv2 as cv
import pandas as pd


# Generate dataset file from images
files_path = 'dataset/English/Fnt/'
# this line count how many files there is in the path
total_folders = len([name for name in os.listdir(files_path) if os.path.isdir(os.path.join(files_path, name))])

def create_csv_from_img(chunk_size=100, debug=False):

    for dir in range(1, total_folders +1):
        dir_path = 'dataset/English/Fnt/Sample' + str(f"{dir:03d}") +'/'
        csv_path = files_path +'char' + str(f"{dir:03d}") + '.csv'
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

        file_size = os.path.getsize(csv_path)
        print(str(file_size >> 10) + ' kb')  # small trick to convert bytes to kb

    return 0

def create_dataset(training_samp_by_char=100,random=True, debug=False):
    dataset_path = 'dataset/' + 'train_datas' + '.csv'
    labels_path = 'dataset/' + 'labels' + '.csv'
    #remove train file if exists
    if os.path.isfile(dataset_path):
        os.remove(dataset_path)
    #create new train file
    with open(dataset_path,'w'):
        pass

    # create new labels file
    if os.path.isfile(labels_path):
        os.remove(labels_path)
    with open(labels_path,'w'):
        pass

    #Here I use total_folder as there's as many csv as folders
    for file in range(1,total_folders+1):
        csv_path = files_path + 'char' + str(f"{file:03d}") + '.csv'
        # if a csv file is missing, abort function and delete train set
        if not(os.path.isfile(csv_path)):
            return -1

        datas = np.zeros((0, 784))
        samples_indexes = np.arange(training_samp_by_char)
        if random:
            samples_indexes = np.random.randint(0,1016,training_samp_by_char)
            all_indexes = np.arange(0,1015)
            skip_indexes = np.delete(all_indexes,samples_indexes)
            df1 = pd.read_csv(csv_path, skiprows=skip_indexes)
        else:
            df1 = pd.read_csv(csv_path, rows=samples_indexes)
        labels = np.ones(training_samp_by_char,dtype=int)*file
        df2 = pd.DataFrame(labels)

        df1.to_csv(dataset_path, index=False, header=False, mode='a', chunksize=training_samp_by_char)
        df2.to_csv(labels_path, index=False, header=False, mode='a', chunksize=training_samp_by_char)
        print(str(file) + '/62 char treated')
    return 0

create_csv_from_img(100)
create_dataset(200)
