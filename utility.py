import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
import random
from imgaug import augmenters as iaa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


def getName(filePath):
    return filePath.split('\\')[-1]


def importData(path):
    cols = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']

    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=cols)

    # get all the center imgs
    data['Center'] = data['Center'].apply(getName)
    print('Total Center Images Imported:', data.shape[0])
    return data

#balancing data
def regData(data,flag=True):
    numBins = 31
    samples_per_bin = 800
    hist, bins = np.histogram(data['Steering'], numBins, range=[-1,1])
    #global center #####
    #print(bins)
    if flag:
        center = (bins[:-1] + bins[1:])*0.5
        plt.bar(center, hist, width=0.06)
        # plot redundant data
        plt.plot((-1,1), (samples_per_bin, samples_per_bin))
        plt.show()

    #removing redundant data
    removeList = []
    for i in range(numBins):
        binInfoList = []
        for j in range(len(data['Steering'])):
            if data['Steering'][j] >= bins[i] and data['Steering'][j] <= bins[i+1]:
                binInfoList.append(j)
        binInfoList = shuffle(binInfoList)
        binInfoList = binInfoList[samples_per_bin:]
        removeList.extend(binInfoList)
    print('Num of Removed Images: ', len(removeList))
    #remove from data
    data.drop(data.index[removeList], inplace=True)
    print('Remaining Images left: ', len(data))
    # final histogram data
    if flag:
        hist, _ = np.histogram(data['Steering'], numBins, range=[-1,1])
        #center = (bins[:-1] + bins[1:]) * 0.5 ##
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samples_per_bin, samples_per_bin))
        plt.show()

    return data
#splitting up the data for training
def inputData(path,data):
    imgsList = []
    steerList = []

    for i in range(len(data)):
        findData = data.iloc[i]

        imgsList.append(os.path.join(path, 'IMG', findData[0]))
        steerList.append(float(findData[3]))

    imgsList = np.asarray(imgsList)
    steerList = np.asarray(steerList)
    return imgsList, steerList

#augmenting the data, by applying changes to imgs: zoom, brightness, panning, flip
def imgAug(imgList, steer):
    image = mpimg.imread(imgList)
    #panning
    if np.random.rand() < 0.5: #generate random val betw 0 and 1, can have multiple augmentations at once
        pan = iaa.Affine(translate_percent={'x':(-0.1, 0.1), 'y':(-0.1, 0.1)})
        image = pan.augment_image(image)

    #brightness chg
    if np.random.rand() < 0.5:
        bright = iaa.Multiply((0.2,1.2))
        image = bright.augment_image(image) #

    #flip img
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steer = -steer           #since flipping img have to chg steering angle

    # zoom imgs
    if np.random.rand() < 0.5:
        zoom_img = iaa.Affine(scale=(1, 1.2))
        image = zoom_img.augment_image(image)


    return image, steer

# img_result, str_ang = imgAug('test_img.jpg',0)
# plt.imshow(img_result)
# plt.show()

#preprocess imgs - crop out any unnecessary background data, car trees, etc
def img_process(image):
    image = image[60:135, :, :]

    #YUV Colourspace - able to better define lane lines and general path
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3),0)
    image = cv2.resize(image, (200,66)) # size used by nvidia
    image = image / 255

    return image


# send images to model in batches
def dataGen(imgsList, steeringList, batch_sz, flag):
    while True:
        batch_list = []
        batch_steer = []

        for k in range(batch_sz):
            loc = random.randint(0, len(imgsList)-1)
            if flag:
                image, steer = imgAug(imgsList[loc], steeringList[loc])
            else:
                image = mpimg.imread(imgsList[loc])
                steer = steeringList[loc]
            image = img_process(image)
            batch_list.append(image)
            batch_steer.append(steer)

        yield (np.array(batch_list),np.asarray(batch_steer))

 # creating nvidia's model
def genModel():
    model = Sequential()
    model.add(Convolution2D(24,(5,5),(2,2), input_shape=(66,200,3), activation="elu"))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation="elu"))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation="elu"))
    model.add(Convolution2D(64, (3, 3), activation="elu"))
    model.add(Convolution2D(64, (3, 3), activation="elu"))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.compile(Adam(lr=0.0001), loss='mse')

    return model