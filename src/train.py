#To detect Moire ́ patternzs, images are first decomposed using Wavelet decomposition (refer to file '') and trained using multi-input Convolutional neural network. The strength of the proposed CNN model is, it uses the LL intensity image (from the Wavelet decomposition) as a weight parameter for the Moire ́ pattern, thereby approximating the spatial spread of the Moire ́ pattern in the image. Usage of CNN model performs better than frequency thresholding approach as the model is trained considering diverse scenarios and it is able to distinguish between the high frequency of background texture and the Moire ́ pattern.

from matplotlib import pyplot as plt
import numpy as np
import sys
import argparse
from os import listdir
from os.path import isfile, join
from PIL import Image
from sklearn import preprocessing
from skimage import io
from sklearn.model_selection import train_test_split
import os
from mCNN import createModel
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.callbacks import ModelCheckpoint

#constants
WIDTH = 500#384
HEIGHT = 375#512

def scaleData(inp, minimum, maximum):
    minMaxScaler = preprocessing.MinMaxScaler(copy=True, feature_range=(minimum,maximum))
    inp = inp.reshape(-1, 1)
    inp = minMaxScaler.fit_transform(inp)
    
    return inp




# - read positive and negative training data
# - create X and Y from training data


def main(args):
    positiveImagePath = (args.positiveImages)
    negativeImagePath = (args.negativeImages)
    numEpochs = (args.epochs)
    positiveTrainImagePath = args.trainingDataPositive
    negativeTrainImagePath = args.trainingDataNegative

    
    X_LL, X_LH, X_HL, X_HH, X_index, Y, imageCount = readWaveletData(positiveImagePath, negativeImagePath, positiveTrainImagePath, negativeTrainImagePath)
    
    X_LL_train,X_LH_train,X_HL_train,X_HH_train,Y_train,X_LL_test,X_LH_test,X_HL_test,X_HH_test,Y_test = trainTestSplit(X_LL, X_LH, X_HL, X_HH, X_index, Y, imageCount)
    
    model = trainCNNModel(X_LL_train,X_LH_train,X_HL_train,X_HH_train,Y_train,
             X_LL_test,X_LH_test,X_HL_test,X_HH_test,Y_test, numEpochs)
    
    evaluate(model, X_LL_test,X_LH_test,X_HL_test,X_HH_test,Y_test)
    
    


def readAndScaleImage(f, customStr, trainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, sampleVal):
    fileName = (os.path.splitext(f)[0])
    fLL = (f.replace(fileName, fileName + customStr + '_LL')).replace('.jpg','.tiff')
    fLH = (f.replace(fileName, fileName + customStr + '_LH')).replace('.jpg','.tiff')
    fHL = (f.replace(fileName, fileName + customStr + '_HL')).replace('.jpg','.tiff')
    fHH = (f.replace(fileName, fileName + customStr + '_HH')).replace('.jpg','.tiff')
    
    try:
        imgLL = Image.open(join(trainImagePath, fLL))
        imgLH = Image.open(join(trainImagePath, fLH))
        imgHL = Image.open(join(trainImagePath, fHL))
        imgHH = Image.open(join(trainImagePath, fHH))
    except Exception as e:
        print('Error: Couldnt read the file {}. Make sure only images are present in the folder'.format(fileName))
        print('Exception:', e)
        return None
        
    imgLL = np.array(imgLL)
    imgLH = np.array(imgLH)
    imgHL = np.array(imgHL)
    imgHH = np.array(imgHH)
    imgLL = scaleData(imgLL, 0, 1)
    imgLH = scaleData(imgLH, -1, 1)
    imgHL = scaleData(imgHL, -1, 1)
    imgHH = scaleData(imgHH, -1, 1)
    
    imgVector = imgLL.reshape(1, WIDTH*HEIGHT)
    X_LL[sampleIndex, :] = imgVector
    imgVector = imgLH.reshape(1, WIDTH*HEIGHT)
    X_LH[sampleIndex, :] = imgVector
    imgVector = imgHL.reshape(1, WIDTH*HEIGHT)
    X_HL[sampleIndex, :] = imgVector
    imgVector = imgHH.reshape(1, WIDTH*HEIGHT)
    X_HH[sampleIndex, :] = imgVector
    
    Y[sampleIndex, 0] = sampleVal;
    X_index[sampleIndex, 0] = sampleIndex;
    
    return True
    
def readImageSet(imageFiles, trainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, bClass):

    for f in imageFiles:
        ret = readAndScaleImage(f, '', trainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, bClass)
        if ret == True:
            sampleIndex = sampleIndex + 1

        #read 180deg rotated data
        ret = readAndScaleImage(f, '_180', trainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex,bClass)
        if ret == True:
            sampleIndex = sampleIndex + 1

        #read 180deg FLIP data
        ret = readAndScaleImage(f, '_180_FLIP', trainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, bClass)
        if ret == True:
            sampleIndex = sampleIndex + 1
     
    return sampleIndex
            
            
def readWaveletData(positiveImagePath, negativeImagePath, positiveTrainImagePath, negativeTrainImagePath):
    
    # get augmented, balanced training data image files by class
    positiveImageFiles = [f for f in listdir(positiveImagePath) if (isfile(join(positiveImagePath, f)))]
    negativeImageFiles = [f for f in listdir(negativeImagePath) if (isfile(join(negativeImagePath, f)))]

    
    positiveCount = len(positiveImageFiles)*4
    negativeCount = len(negativeImageFiles)*4

    print('positive samples: ' + str(positiveCount))
    print('negative samples: ' + str(negativeCount))
    imageCount = positiveCount + negativeCount
    #intialization
    X_LL = np.zeros((positiveCount + negativeCount, WIDTH*HEIGHT))
    X_LH = np.zeros((positiveCount + negativeCount, WIDTH*HEIGHT))
    X_HL = np.zeros((positiveCount + negativeCount, WIDTH*HEIGHT))
    X_HH = np.zeros((positiveCount + negativeCount, WIDTH*HEIGHT))
    X_index = np.zeros((positiveCount + negativeCount, 1))
    Y = np.zeros((positiveCount + negativeCount, 1))
    
    sampleIndex = 0
    # read all images, convert to float, divide by 255 (leads to gray range 0..1), reshape into a row vector
    # write class 0 for positive and 1 for negative samples    
    sampleIndex = readImageSet(positiveImageFiles, positiveTrainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, 0)
    print('positive data loaded.')
    
    sampleIndex += readImageSet(negativeImageFiles, negativeTrainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, 1)
    print('negative data loaded.')

    print('Total Samples Loaded: ', sampleIndex)
    print(X_LL)
    print(X_LH)
    print(Y)
    
    return X_LL, X_LH, X_HL, X_HH, X_index, Y, imageCount



#Here, we perform index based splitting and use those indices to split the our multi-input datasets. This is done because the CNN model is multi-input network
def splitTrainTestDataForBands(inputData, X_train_ind, X_test_ind):
    X_train = np.zeros((len(X_train_ind), WIDTH*HEIGHT))
    for i in range(len(X_train_ind)):
        X_train[i,:] = inputData[int(X_train_ind[i,0]),:]
        
    X_test = np.zeros((len(X_test_ind), WIDTH*HEIGHT))
    for i in range(len(X_test_ind)):
        X_test[i,:] = inputData[int(X_test_ind[i,0]),:]
        
    return X_train, X_test


def countPositiveSamplesAfterSplit(trainData):
    count = 0;
    for i in range(len(trainData)):
        if(trainData[i,0] == 0):
            count = count + 1
    return count




def trainTestSplit(X_LL, X_LH, X_HL, X_HH, X_index, Y, imageCount):
    testCountPercent = 0.1

    # evaluate the model by splitting into train and test sets
    X_train_ind, X_test_ind, y_train, y_test = train_test_split(X_index, Y, test_size=testCountPercent, random_state=1, stratify=Y)

    X_LL_train, X_LL_test = splitTrainTestDataForBands(X_LL, X_train_ind, X_test_ind)
    X_LH_train, X_LH_test = splitTrainTestDataForBands(X_LH, X_train_ind, X_test_ind)
    X_HL_train, X_HL_test = splitTrainTestDataForBands(X_HL, X_train_ind, X_test_ind)
    X_HH_train, X_HH_test = splitTrainTestDataForBands(X_HH, X_train_ind, X_test_ind)

    imageHeight = HEIGHT
    imageWidth = WIDTH


    print(countPositiveSamplesAfterSplit(y_train))
    print(len(X_LL_train))
    print(len(y_train))
    print(len(X_LL_test))
    print(len(y_test))

    num_train_samples = len(y_train)
    print('num_train_samples', num_train_samples)
    X_LL_train = np.array(X_LL_train)
    X_LL_train = X_LL_train.reshape((num_train_samples, imageHeight, imageWidth, 1))
    X_LL_test = np.array(X_LL_test)
    X_LL_test = X_LL_test.reshape((imageCount - num_train_samples, imageHeight, imageWidth, 1))

    X_LH_train = np.array(X_LH_train)
    X_LH_train = X_LH_train.reshape((num_train_samples, imageHeight, imageWidth, 1))
    X_LH_test = np.array(X_LH_test)
    X_LH_test = X_LH_test.reshape((imageCount - num_train_samples, imageHeight, imageWidth, 1))

    X_HL_train = np.array(X_HL_train)
    X_HL_train = X_HL_train.reshape((num_train_samples, imageHeight, imageWidth, 1))
    X_HL_test = np.array(X_HL_test)
    X_HL_test = X_HL_test.reshape((imageCount - num_train_samples, imageHeight, imageWidth, 1))
    
    X_HH_train = np.array(X_HH_train)
    X_HH_train = X_HH_train.reshape((num_train_samples, imageHeight, imageWidth, 1))
    X_HH_test = np.array(X_HH_test)
    X_HH_test = X_HH_test.reshape((imageCount - num_train_samples, imageHeight, imageWidth, 1))

    y_train = np.array(y_train)
    y_test = np.array(y_test)


    num_train, height, width, depth = X_LL_train.shape
    num_test = X_LL_test.shape[0] 
    num_classes = len(np.unique(y_train))
    
    return X_LL_train,X_LH_train,X_HL_train,X_HH_train,y_train,X_LL_test,X_LH_test,X_HL_test,X_HH_test,y_test




def trainCNNModel(X_LL_train,X_LH_train,X_HL_train,X_HH_train,y_train,
             X_LL_test,X_LH_test,X_HL_test,X_HH_test,y_test, num_epochs):

    batch_size = 32 # in each iteration, we consider 32 training examples at once
    num_train, height, width, depth = X_LL_train.shape
    num_classes = len(np.unique(y_train))
    Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
    Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

    checkPointFolder = 'checkPoint'
    checkpoint_name = checkPointFolder + '/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    callbacks_list = [checkpoint]
    
    if not os.path.exists(checkPointFolder):
        os.makedirs(checkPointFolder)
        
        
    model = createModel(height, width, depth, num_classes)
    
    model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
                  optimizer='adam', # using the Adam optimiser
                  metrics=['accuracy']) # reporting the accuracy

    model.fit([X_LL_train,X_LH_train,X_HL_train,X_HH_train], Y_train,                # Train the model using the training set...
              batch_size=batch_size, epochs=num_epochs,
              verbose=1, validation_split=0.1, callbacks=callbacks_list) # ...holding out 10% of the data for validation
    score, acc = model.evaluate([X_LL_test,X_LH_test,X_HL_test,X_HH_test], Y_test, verbose=1)  # Evaluate the trained model on the test set!

    model.save('moirePattern3CNN_.h5')
    
    return model


def evaluate(model, X_LL_test,X_LH_test,X_HL_test,X_HH_test,y_test):

    model_out = model.predict([X_LL_test,X_LH_test,X_HL_test,X_HH_test])
    passCnt = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y_test)):
        if np.argmax(model_out[i, :]) == y_test[i]:
            str_label='Pass'
            passCnt = passCnt + 1
        else:
            str_label='Fail'

        if y_test[i] ==0:
            if np.argmax(model_out[i, :]) == y_test[i]:
                TP = TP + 1;
            else:
                FN = FN + 1
        else:
            if np.argmax(model_out[i, :]) == y_test[i]:
                TN = TN + 1;
            else:
                FP = FP + 1

    start = "\033[1m"
    end = "\033[0;0m"
    print(start + 'confusion matrix (test / validation)' + end)
    print(start + 'true positive:  '+ end + str(TP))
    print(start + 'false positive: '+ end + str(FP))
    print(start + 'true negative:  '+ end + str(TN))
    print(start + 'false negative: '+ end + str(FN))
    print('\n')
    print(start + 'accuracy:  ' + end + "{:.4f} %".format(100*(TP+TN)/(TP+FP+FN+TN)))
    print(start + 'precision: ' + end + "{:.4f} %".format(100*TP/(TP + FP)))
    print(start + 'recall:  ' + end + "{:.4f} %".format(100*TP/(TP + FN)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('positiveImages', type=str, help='Directory with original positive (Moiré pattern) images.')
    parser.add_argument('negativeImages', type=str, help='Directory with original negative (Normal) images.')
    
    parser.add_argument('trainingDataPositive', type=str, help='Directory with transformed positive (Moiré pattern) images.')
    parser.add_argument('trainingDataNegative', type=str, help='Directory with transformed negative (Normal) images.')
    
    parser.add_argument('epochs', type=int, help='Number of epochs for training')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    
    