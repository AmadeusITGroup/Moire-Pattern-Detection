# Moire Pattern Detection


## Introduction
To detect Moire ́ patterns, images are first decomposed using Wavelet decomposition and trained using multi-input Convolutional neural network. The strength of the proposed CNN model is, it uses the LL intensity image (from the Wavelet decomposition) as a weight parameter for the Moire ́ pattern, thereby approximating the spatial spread of the Moire ́ pattern in the image. Usage of CNN model performs better than frequency thresholding approach as the model is trained considering diverse scenarios and it is able to distinguish between the high frequency of background texture and the Moire ́ pattern.

If this code helps with your work, please cite:

```bibtex
@INPROCEEDINGS{8628746,
author={E. {Abraham}},
booktitle={2018 IEEE Symposium Series on Computational Intelligence (SSCI)},
title={Moiré Pattern Detection using Wavelet Decomposition and Convolutional Neural Network},
year={2018},
volume={},
number={},
pages={1275-1279},
ISSN={},
month={Nov},}
```

## Set-up



### 1. Install Python3 

### 2. Install dependencies 

`pip install tensorflow`

`pip install keras`

`pip install Pillow`


### 3. create the wavelet decomposed training dataset from the captured images

`python createTrainingData.py positiveImages negativeImages train`

```
positional arguments:

  positiveImages  Directory with positive (Moiré pattern) images.
  
  negativeImages  Directory with negative (Normal) images.
  
  train           0 = train, 1 = test
```

### 4. train the CNN model using training images

`python train.py positiveImages negativeImages trainingDataPositive trainingDataNegative epochs`

```
positional arguments:

  positiveImages        Directory with original positive (Moiré pattern)
                        images.
                        
  negativeImages        Directory with original negative (Normal) images.
  
  trainingDataPositive  Directory with transformed positive (Moiré pattern)
                        images.
                        
  trainingDataNegative  Directory with transformed negative (Normal) images.
  
  epochs                Number of epochs for training
```

### 5. test the CNN model using test images

`python test.py moirePattern3CNN_.h5  positiveImages negativeImages`

```
positional arguments:

  weightsFile               saved CNN model file
  
  positiveTestImages        Directory with original positive (Moiré pattern)
                        images.
                        
  negativeTestImages        Directory with original negative (Normal) images.
  

```




## Documentation
IEEE SSCI-2018 publication (https://ieeexplore.ieee.org/document/8628746).

Note: The paper shows 3 bands of wavelet decomposition images taken as input to CNN as compared to the 4 bands in this python implementation





## Additional information
If you have any technical questions, feel free to [contact](mailto:eldho.abraham@amadeus.com) us or create an issue [here](https://linkToBeProvided.com).
