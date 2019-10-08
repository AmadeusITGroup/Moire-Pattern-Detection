#Use this file for evaluating on a dataset that is not used for training

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
import createTrainingData
from train import readWaveletData, evaluate

#constants
width = 500#384 #change dimensions according to the input image in the training
height = 375#512 #change dimensions according to the input image in the training
depth = 1
num_classes = 2

positiveTestImagePath = './testDataPositive'
negativeTestImagePath = './testDataNegative'
    
def main(args):
    weights_file = (args.weightsFile)
    positiveImagePath = (args.positiveTestImages)
    negativeImagePath = (args.negativeTestImages)
    
    os.system("python createTrainingData.py {} {} {}".format(positiveImagePath, negativeImagePath, 1))
    X_LL, X_LH, X_HL, X_HH, X_index, Y, imageCount = readWaveletData(positiveImagePath, negativeImagePath, positiveTestImagePath, positiveTestImagePath)
    
    X_LL = np.array(X_LL)
    X_LL = X_LL.reshape((imageCount, height, width, depth))
    X_LH = np.array(X_LH)
    X_LH = X_LH.reshape((imageCount, height, width, depth))
    X_HL = np.array(X_HL)
    X_HL = X_HL.reshape((imageCount, height, width, depth))
    X_HH = np.array(X_HH)
    X_HH = X_HH.reshape((imageCount, height, width, depth))
    
    CNN_model = createModel(height, width, depth, num_classes)
    CNN_model.load_weights(weights_file)
    evaluate(CNN_model,X_LL,X_LH,X_HL,X_HH, Y)



def run(model, X_LL_test,X_LH_test,X_HL_test,y_test):
    return


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('weightsFile', type=str, help='saved CNN model file')
    
    parser.add_argument('positiveTestImages', type=str, help='Directory with positive (Moiré pattern) images.')
    parser.add_argument('negativeTestImages', type=str, help='Directory with negative (Normal) images.')
    
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    
    