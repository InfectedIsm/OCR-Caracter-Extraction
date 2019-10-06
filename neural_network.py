#=========================================
#Keras imports
from keras import models as kerasModels
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

#OpenCV and MatPlotLib imports
import cv2
import matplotlib.pyplot as plt

#Local application imports
from manage_dataset import *

#=========================================

#Local Variables
MODEL_PATH = "models/CharRecognition.model"
#Pre_trained model from TrainModel() function
MODEL = kerasModels.load_model(MODEL_PATH)
MODEL_INPUT_SHAPE = (1, 28, 28, 1)

#=========================================

def UseModel(image, model=MODEL):

    indexes_to_char = IndexToChar()
    img = np.resize(image,(28,28))
    to_predict = np.reshape(img, (1, 28, 28, 1))
    prediction = model.predict(to_predict)

    char_predicted = indexes_to_char[np.argmax(prediction)]

    return char_predicted

def SaveCharToImg(img, path):
    img = np.reshape(img, (28, 28)) * 255
    img = cv2.resize(img, (280, 280))
    path = str(path)
    print(cv2.imwrite(path, img))

def TestModelInteractive(test_datas, test_labels,
                         number_of_tests=5, infinite_loop=False, plot=True):

    # variable 'first_call' is necessary to enter in the while loop even if variable
    # infinite_loop = False
    first_call = True
    n_errors = 0
    total_test = 0
    error_list = {}

    indexes_to_char = IndexToChar()

    while (infinite_loop or first_call):
        for i in range(number_of_tests):
            index = np.random.randint(np.shape(test_datas)[0]-1)

            img = test_datas[index, :, :, :]
            # to_test = np.reshape(img, MODEL_INPUT_SHAPE)
            #
            # prediction = model.predict(to_test)
            true_label = test_labels[index]
            #
            # char_predicted = indexes_to_char[np.argmax(prediction)]

            char_predicted = UseModel(img)
            char_true_label = indexes_to_char[np.argmax(true_label)]

            if (char_predicted != char_true_label):
                n_errors = n_errors + 1
                error_description = str(char_predicted + "-but-" + char_true_label)
                path= str('dataset/FailedChar/' + error_description + '_(' + str(i) + ').png')
                SaveCharToImg(img,path)
                if error_description in error_list:
                    error_list[error_description] += 1
                else:
                    error_list[error_description] = 1

            print("predicted: " + char_predicted +
                  "  true: " + char_true_label)

            total_test += 1

            if plot:
                img = np.reshape(img, (28, 28))*255
                fig = plt.figure(0)
                ax = fig.add_subplot(111)
                plt.imshow(img, cmap=plt.get_cmap('gray'))
                plt.title('model loss')
                plt.show()

        first_call=False

        again = input("Restart for " + str(number_of_tests) + "char? [y/n]:")
        if again==("y" or "Y"):
            continue
        else:
            accuracy = 1-(n_errors / total_test)
            print('accuracy' + str(accuracy))
            break

    return error_list

#You can remove this line if the dataset is already existing
# CreateDataset(1000)

def GenerateTrainTestDatas(train_percentage = 0.95):
    dataset_percentage = 0.1
    train_percentage = 0.95

    x, y = CsvToArray(percentage=1)
    x = x / 255
    y = y

    dataset_shape = np.shape(x)
    train_indexes = GenerateRandomIndexes(size=int(dataset_shape[0] * train_percentage),
                                          max=dataset_shape[0])
    test_indexes = np.delete(np.arange(dataset_shape[0]), train_indexes)

    x_train = np.array(x[train_indexes, :])
    x_train = np.reshape(x_train, (np.shape(x_train)[0], 28, 28, 1))
    y_train = np.array(y[train_indexes, :]) - 1
    # to_categorical function turn an int into a one hot encoded vector
    y_train = utils.to_categorical(y_train, num_classes=62)

    x_test = np.array(x[test_indexes, :])
    x_test = np.reshape(x_test, (np.shape(x_test)[0], 28, 28, 1))
    y_test = np.array(y[test_indexes, :]) - 1

    y_test = utils.to_categorical(y_test, num_classes=62)

    return x_train, y_train, x_test, y_test

def TrainModel(save=True):

    x_train, x_test, y_train, y_test = GenerateTrainTestDatas()

    model = Sequential()
    # input: 28x28 images with 1 channels (no colors) -> (100, 100, 1) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(24, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))

    model.add(Conv2D(12, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Flatten())
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(62, activation='softmax'))

    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=100, epochs=15, validation_split=0.05)
    score = model.evaluate(x_test, y_test, batch_size=100)

    if save==True:
        model.save(MODEL_PATH)

    # model.summary()

    print(model.metrics_names)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100) )
    cvscores = []
    cvscores.append(score[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    # print(history.history.keys())
    # summarize history for accuracy

    # summarize history for loss
    plt.figure()
    plt.subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    axes = plt.gca()
    axes.set_yscale('log')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.subplot(122)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    axes = plt.gca()
    axes.set_yscale('log')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()

    return 0


# _,_,x_test,y_test = GenerateTrainTestDatas()
#
# error_list= TestModelInteractive(x_test, y_test,
#                                 number_of_tests=2000,
#                                 infinite_loop=False,
#                                 plot=False)
#
# print(error_list)