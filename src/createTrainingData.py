import sys
import argparse
from PIL import Image
from PIL import ImageOps
import random
import sys
import os

from os import listdir
from os.path import isfile, join
from PIL import Image
from haar2D import fwdHaarDWT2D

#The training images need to be put in two folders. positiveImages and negativeImages. positiveImages are the images which are captured from the display devices and has the presence of stron or weak Moiré patterms in it. negativeImages are the ones without Moiré Patterns (i.e. the images which are not captured from the display devices)


#folders to store training data
positiveTrainImagePath = ''
negativeTrainImagePath = ''

def main(args):
    
    global positiveTrainImagePath
    global negativeTrainImagePath
    
    positiveImagePath = (args.positiveImages)
    negativeImagePath = (args.negativeImages)
    
    if (args.train == 0):
        positiveTrainImagePath = './trainDataPositive'
        negativeTrainImagePath = './trainDataNegative'
    else:
        positiveTrainImagePath = './testDataPositive'
        negativeTrainImagePath = './testDataNegative'
        
    createTrainingData(positiveImagePath, negativeImagePath)

    
#The wavelet decomposed images are the transformed images representing the spatial and the frequency information of the image. These images are stored as 'tiff' in the disk, to preserve that information. Each image is transformed with 180 degrees rotation and as well flipped, as part of data augmentation.

def transformImageAndSave(image, f, customStr, path):
    cA, cH, cV, cD  = fwdHaarDWT2D(image);
    
    fileName = (os.path.splitext(f)[0])
    fLL = (f.replace(fileName, fileName+'_' + customStr + 'LL')).replace('.jpg','.tiff')
    fLH = (f.replace(fileName, fileName+'_' + customStr + 'LH')).replace('.jpg','.tiff')
    fHL = (f.replace(fileName, fileName+'_' + customStr + 'HL')).replace('.jpg','.tiff')
    fHH = (f.replace(fileName, fileName+'_' + customStr + 'HH')).replace('.jpg','.tiff')
    cA = Image.fromarray(cA)
    cH = Image.fromarray(cH)
    cV = Image.fromarray(cV)
    cD = Image.fromarray(cD)
    cA.save(join(path, fLL))
    cH.save(join(path, fLH))
    cV.save(join(path, fHL))
    cD.save(join(path, fHH))
    
    
def augmentAndTrasformImage(f, mainFolder, trainFolder):
    try:
        img = Image.open(join(mainFolder, f)) 
    except:
        print('Error: Couldnt read the file {}. Make sure only images are present in the folder'.format(f))
        return None

    imgGray = img.convert('L')
    wdChk, htChk = imgGray.size
    if htChk > wdChk:
        imgGray = imgGray.rotate(-90, expand=1)
        print('training image rotated')
    transformImageAndSave(imgGray, f, '', trainFolder)

    imgGray = imgGray.transpose(Image.ROTATE_180)
    transformImageAndSave(imgGray, f, '180_', trainFolder)

    imgGray = imgGray.transpose(Image.FLIP_LEFT_RIGHT)
    transformImageAndSave(imgGray, f, '180_FLIP_', trainFolder)
    
    return True
    
    
def createTrainingData(positiveImagePath, negativeImagePath):
    
    # get image files by classes
    positiveImageFiles = [f for f in listdir(positiveImagePath) if (isfile(join(positiveImagePath, f)))]
    negativeImageFiles = [f for f in listdir(negativeImagePath) if (isfile(join(negativeImagePath, f)))]

    positiveCount = len(positiveImageFiles)
    negativeCount = len(negativeImageFiles)

    print('positive samples: ' + str(positiveCount))
    print('negative samples: ' + str(negativeCount))
    
    # create folders (not tracked by git)
    if not os.path.exists(positiveTrainImagePath):
        os.makedirs(positiveTrainImagePath)
    if not os.path.exists(negativeTrainImagePath):
        os.makedirs(negativeTrainImagePath)

    Knegative = 0
    Kpositive = 0

    # create positive training images 
    for f in positiveImageFiles:
        ret = augmentAndTrasformImage(f, positiveImagePath, positiveTrainImagePath)
        if ret == None:
            continue
        Kpositive += 3


    # create negative training images 
    for f in negativeImageFiles:
        ret = augmentAndTrasformImage(f, negativeImagePath, negativeTrainImagePath)
        if ret == None:
            continue
        Knegative += 3;
    
    print('Total positive files after augmentation: ', Kpositive)
    print('Total negative files after augmentation: ', Knegative)
    
        

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('positiveImages', type=str, help='Directory with positive (Moiré pattern) images.')
    parser.add_argument('negativeImages', type=str, help='Directory with negative (Normal) images.')
    parser.add_argument('train', type=int, help='0 = train, 1 = test')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

          
