import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

import numpy as np
from manage_dataset import *

import cv2
import matplotlib.pyplot as plt


def TestModelInteractive(test_datas, number_of_tests=5):
    for i in range(number_of_tests):
        index = np.random.randint(1000)

        img = test_datas[index, :, :, :]
        to_test = np.reshape(img, (1, 28, 28, 1))
        prediction = model.predict(to_test)
        print(np.argmax(prediction))

        img = np.reshape(img, (28, 28))*25.5-1
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.show()


#You can remove this line if the dataset is already existing
#CreateDataset(500)

dataset_percentage = 1
train_percentage = 0.9

x,y=CsvToArray(percentage=1)
x = x/25.5+1
y = y/25.5+1


dataset_shape = np.shape(x)
train_indexes = GenerateRandomIndexes(int(dataset_shape[0]*train_percentage),dataset_shape[0])
test_indexes = np.delete(np.arange(dataset_shape[0]),train_indexes)

x_train = np.array(x[train_indexes,:])
x_train = np.reshape(x_train,(np.shape(x_train)[0],28,28,1))
y_train = np.array(y[train_indexes,:])-1
# to_categorical function turn an int into a one hot encoded vector
y_train = keras.utils.to_categorical(y_train, num_classes=62)

np.concatenate

x_test = np.array(x[test_indexes,:])
x_test = np.reshape(x_test,(np.shape(x_test)[0],28,28,1))
y_test = np.array(y[test_indexes,:])-1
y_test = keras.utils.to_categorical(y_test, num_classes=62)

# for i in range(10):
#     index = np.random.randint(1000)
#
#     img = x_train[index,:,:,:]
#     img = np.reshape(img,(28,28))
#     print(y_train[index])
#
#     fig = plt.figure(0)
#     ax = fig.add_subplot(111)
#     plt.imshow(img, cmap=plt.get_cmap('gray'))
#     plt.show()



model = Sequential()
# input: 28x28 images with 1 channels (no colors) -> (100, 100, 1) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Conv2D(16, (3, 3), activation='relu'))
# model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(62, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=100, epochs=100)
score = model.evaluate(x_test, y_test, batch_size=32)

model.summary()

# print(model.metrics_names)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100) )
# cvscores = []
# cvscores.append(score[1] * 100)
# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# print(history.history.keys())
# summarize history for accuracy

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
axes = plt.gca()
axes.set_yscale('log')
axes.set_xlim([2,10])
axes.set_ylim([0.1,1e-5])

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


TestModelInteractive(x_test,10)

