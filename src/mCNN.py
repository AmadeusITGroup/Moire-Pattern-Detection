
#The convolutional layer C1 filter three 512 x 384 input images with 32 kernels of size 7 x 7 with a stride of 1 pixel. The stride of pooling layer S1 is 2 pixels. Then, the convolved images of LH and HL are merged together by taking the maximum from both the images. In the next step, the convolved image of LL is merged with the Max result by multiplying both the results (as explained in section III-B). C2-C4 has 16 kernels of size 3 x 3 with a stride of 1 pixel. S2 pools the merged features with a stride of 4. The dropout is applied to the output of S4 which has been flattened. The fully connected layer FC1 has 32 neurons and FC2 has 1 neuron. The activation of the output layer is a softmax function.

#One significant observation found while analyzing the images is, when the image intensities are relatively darker, the patterns aren’t visible much. The darker regions in the image produce less Moire ́ patterns compared to the brighter regions in the image. To summarize the spread of the Moire ́ pattern in the image, spatially, and to produce this effect while training the network, we used the LL band of the image (which is the downsampled original image consisting of low frequency information) and used it as weights for LH anf HL band during the training, by directly multiplying it to the convolved and combined response of the LH and HL bands

import os

from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Add, Multiply, Maximum

def createModel(height, width, depth, num_classes):
#     num_epochs = 20 # 50 26 200 # we iterate 200 times over the entire training set
    kernel_size_1 = 7 # we will use 7x7 kernels 
    kernel_size_2 = 3 # we will use 3x3 kernels 
    pool_size = 2 # we will use 2x2 pooling throughout
    conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
    conv_depth_2 = 16 # ...switching to 16 after the first pooling layer
    drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
    drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
    hidden_size = 32 # 128 512 the FC layer will have 512 neurons


    inpLL = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)
    inpLH = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)
    inpHL = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)
    inpHH = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)

    conv_1_LL = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpLL)
    conv_1_LH = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpLH)
    conv_1_HL = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpHL)
    conv_1_HH = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpHH)
    pool_1_LL = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_LL)
    pool_1_LH = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_LH)
    pool_1_HL = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_HL)
    pool_1_HH = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_HH)

    avg_LH_HL_HH = Maximum()([pool_1_LH, pool_1_HL, pool_1_HH])
    inp_merged = Multiply()([pool_1_LL, avg_LH_HL_HH])
    C4 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(inp_merged)
    S2 = MaxPooling2D(pool_size=(4, 4))(C4)
    drop_1 = Dropout(drop_prob_1)(S2)
    C5 = Convolution2D(conv_depth_1, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(drop_1)
    S3 = MaxPooling2D(pool_size=(pool_size, pool_size))(C5)
    C6 = Convolution2D(conv_depth_1, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(S3)
    S4 = MaxPooling2D(pool_size=(pool_size, pool_size))(C6)
    drop_2 = Dropout(drop_prob_1)(S4)
    # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(num_classes, activation='softmax')(drop_3)
    
    model = Model(inputs=[inpLL, inpLH, inpHL, inpHH], outputs=out) # To define a model, just specify its input and output layers
    
    return model

