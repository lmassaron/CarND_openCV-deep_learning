
## Luca Massaron: Model for Behavioural Cloning based on the paper "End to End Learning for Self-Driving Cars" by NVIDIA engineers.


```python
import numpy as np
import pandas as pd
import json
import re
import os
from scipy import ndimage, misc

import cv2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop, adagrad
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
```

    Using Theano backend.
    Using gpu device 0: GeForce GTX 960M (CNMeM is enabled with initial size: 75.0% of memory, cuDNN 5103)
    

### Attention, this model has been built with Theano, so in the drive.py script needs to modify the structure of images. Please use the provided drive.py. The problem with Theano is with convnets which expect the color channels first. With Tensorflow the color channels are expected as the last dimension in the array.


```python
# I set the Keras backend to Theano, in fact this model has been built on a Windows machine, running Theano since working on 
# a Linux virtual machine proved impossible. The model can run on both Theano and Tensorflow backend, both on Windows or Linux
# just by some slight modification to the structure of images
from keras import backend as K
K.set_image_dim_ordering('th')
```

### Discussion about the neural network architecture

Based on the paper "End to End Learning for Self Driving Cars", I've built a sequence of 5 convnets, expecting array images of size 3x160x320, whose values are normalized in the range [0-1]:

1. the first  has convolution filters of 3 ,  5x5 kernel, and 2x2 strides
2. the second has convolution filters of 24,  5x5 kernel, and 2x2 strides
3. the third  has convolution filters of 36,  5x5 kernel, and 2x2 strides
4. the forth  has convolution filters of 48,  3x3 kernel, and 1x1 strides
5. the fifth  has convolution filters of 64,  3x3 kernel, and 1x1 strides
6. At the end there is a maxpooling with pool size 3x3
7. After the maxpooling there is a dropout fixed to 0.2 (20% of weights to be dropped)

Each convolution layer has a ReLu Activation (to allow non linearity in the model) and before the activation a batchnormalization for each color channel.
batchnormalization normalizes the results after each convnet resulting in speedier at a larger learning rate especially for SGD optimization. It also makes the learning process more robust.
The benefits and characteristcs of batchnormalization are described in this paper: https://arxiv.org/abs/1502.03167
As dropout, also batchnormalization regularizes the weights and allows better generalization (less overfitting)

After this long convnet, the results are flattened and there are three hidden layers of 150, 50 and 10 nodes. Each layer has batchnormalization, ReLu activation ((to allow non linearity in the model, but the last one has linear activation), and dropout 0.2. Based on same examples of networks built for regression, I've tried to initialize them using the normal distribution, not the uniform one (I do not know if it has been beneficial or irrilevant, anyway).

The last hidden layer has linear activation because I thought that ReLu would have forced the network to arrive at the end with just positive or zero values, thus relying on negative bias. Using linear activation I found the network to work better likely because it can have both positive and negative weights and blend them together in the last output node which is the prediction.

Since it is an optimization problem I used Mean Squared Error for loss. I used SGD and set the parameters accordingly to what commonly suggested when using batchnormalization: high learning rate (0.1), low decay rate, high momentum (0.9 but they also suggest 0.99: I found 0.9 working much better) and Nesterov accelerated gradient (NAG).



```python
from IPython.display import Image
Image(filename='architecture.jpg') 
```




![jpeg](output_5_0.jpeg)




```python
def ETE(weights_path=None, color_type_global=3, img_rows=160, img_cols=320):
    """
    Keras' neural architecture mimicking the solution illustrated in the paper "End to End Learning for Self Driving Cars"
    """
    # Size of filters
    nb_filters = 24
    nb_filters = 36
    nb_filters = 48
    nb_filters = 64
    # size of pooling area for max pooling
    pool_size = (3, 3)
    # convolution kernel size
    input_shape = (color_type_global,img_rows,img_cols)

    model = Sequential()
    # First normalization layer
    model.add(BatchNormalization(axis=1, input_shape=input_shape))
    
    # Convolutional layers made of 5 convolutions, a maxpooling and a dropout
    # Each layer has ReLu activations and Batchnormalization
    model.add(Convolution2D(3, 5, 5,border_mode='valid', subsample=(2, 2)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 3, 3, border_mode='valid', subsample=(1, 1)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))
    
    # Flattening of the neurons, first hidden layer with ReLu activation, batchnormalization and dropout
    model.add(Flatten())
    model.add(Dense(150, init='normal')) # W_constraint=maxnorm(3)
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))    
    model.add(Dropout(0.2))
    
    # Second hidden layer with ReLu activation, batchnormalization and dropout
    model.add(Dense(50, init='normal'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))    
    model.add(Dropout(0.2))
    
    # Third hidden layer with ReLu activation, batchnormalization and dropout
    model.add(Dense(10, init='normal'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('linear'))
    
    # Output neuron
    model.add(Dense(1, name="output"))

    if weights_path:
        # If necessary I can load any pre-computed weights
        model.load_weights(weights_path)

    print(model.summary())

    # Optimization is SDG, with high learning rate (allowed by batchnormalization)
    # The parameters have been set inspired by the paper:
    # "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
    
    optim = SGD(lr=0.1, decay=1e-8, momentum=0.9, nesterov=True)

    # Since it is a regression problem the 
    model.compile(loss = 'mean_squared_error',
                  optimizer = optim,
                  metrics = ['mse'])
    return model
```


```python
# The input shape of the images is defined and feed into the model which is compiled
# Please notice that we are working with Theano, so the color channels are put before the width and height
# This implies that we have to modify any image feed into the net accordingly to this schema
color_type_global = 3
img_rows, img_cols = 160, 320
model = ETE(None, color_type_global, img_rows, img_cols)
```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    batchnormalization_1 (BatchNormal(None, 3, 160, 320)   6           batchnormalization_input_1[0][0] 
    ____________________________________________________________________________________________________
    convolution2d_1 (Convolution2D)  (None, 3, 78, 158)    228         batchnormalization_1[0][0]       
    ____________________________________________________________________________________________________
    batchnormalization_2 (BatchNormal(None, 3, 78, 158)    6           convolution2d_1[0][0]            
    ____________________________________________________________________________________________________
    activation_1 (Activation)        (None, 3, 78, 158)    0           batchnormalization_2[0][0]       
    ____________________________________________________________________________________________________
    convolution2d_2 (Convolution2D)  (None, 24, 37, 77)    1824        activation_1[0][0]               
    ____________________________________________________________________________________________________
    batchnormalization_3 (BatchNormal(None, 24, 37, 77)    48          convolution2d_2[0][0]            
    ____________________________________________________________________________________________________
    activation_2 (Activation)        (None, 24, 37, 77)    0           batchnormalization_3[0][0]       
    ____________________________________________________________________________________________________
    convolution2d_3 (Convolution2D)  (None, 36, 17, 37)    21636       activation_2[0][0]               
    ____________________________________________________________________________________________________
    batchnormalization_4 (BatchNormal(None, 36, 17, 37)    72          convolution2d_3[0][0]            
    ____________________________________________________________________________________________________
    activation_3 (Activation)        (None, 36, 17, 37)    0           batchnormalization_4[0][0]       
    ____________________________________________________________________________________________________
    convolution2d_4 (Convolution2D)  (None, 48, 15, 35)    15600       activation_3[0][0]               
    ____________________________________________________________________________________________________
    batchnormalization_5 (BatchNormal(None, 48, 15, 35)    96          convolution2d_4[0][0]            
    ____________________________________________________________________________________________________
    activation_4 (Activation)        (None, 48, 15, 35)    0           batchnormalization_5[0][0]       
    ____________________________________________________________________________________________________
    convolution2d_5 (Convolution2D)  (None, 64, 13, 33)    27712       activation_4[0][0]               
    ____________________________________________________________________________________________________
    batchnormalization_6 (BatchNormal(None, 64, 13, 33)    128         convolution2d_5[0][0]            
    ____________________________________________________________________________________________________
    activation_5 (Activation)        (None, 64, 13, 33)    0           batchnormalization_6[0][0]       
    ____________________________________________________________________________________________________
    maxpooling2d_1 (MaxPooling2D)    (None, 64, 4, 11)     0           activation_5[0][0]               
    ____________________________________________________________________________________________________
    dropout_1 (Dropout)              (None, 64, 4, 11)     0           maxpooling2d_1[0][0]             
    ____________________________________________________________________________________________________
    flatten_1 (Flatten)              (None, 2816)          0           dropout_1[0][0]                  
    ____________________________________________________________________________________________________
    dense_1 (Dense)                  (None, 150)           422550      flatten_1[0][0]                  
    ____________________________________________________________________________________________________
    batchnormalization_7 (BatchNormal(None, 150)           300         dense_1[0][0]                    
    ____________________________________________________________________________________________________
    activation_6 (Activation)        (None, 150)           0           batchnormalization_7[0][0]       
    ____________________________________________________________________________________________________
    dropout_2 (Dropout)              (None, 150)           0           activation_6[0][0]               
    ____________________________________________________________________________________________________
    dense_2 (Dense)                  (None, 50)            7550        dropout_2[0][0]                  
    ____________________________________________________________________________________________________
    batchnormalization_8 (BatchNormal(None, 50)            100         dense_2[0][0]                    
    ____________________________________________________________________________________________________
    activation_7 (Activation)        (None, 50)            0           batchnormalization_8[0][0]       
    ____________________________________________________________________________________________________
    dropout_3 (Dropout)              (None, 50)            0           activation_7[0][0]               
    ____________________________________________________________________________________________________
    dense_3 (Dense)                  (None, 10)            510         dropout_3[0][0]                  
    ____________________________________________________________________________________________________
    batchnormalization_9 (BatchNormal(None, 10)            20          dense_3[0][0]                    
    ____________________________________________________________________________________________________
    activation_8 (Activation)        (None, 10)            0           batchnormalization_9[0][0]       
    ____________________________________________________________________________________________________
    output (Dense)                   (None, 1)             11          activation_8[0][0]               
    ====================================================================================================
    Total params: 498397
    ____________________________________________________________________________________________________
    None
    


```python
def rotate(image_data):
    """
    Rotates the channels in front of the height and width
    """
    return np.swapaxes(np.swapaxes(image_data,2,3),1,2)

def flip_horizontally(image):
    """
    Flips an image horizontally
    """
        return cv2.flip(image,1)
    
def chunks(l, n):
    """
    Yield successive n-sized chunks from l
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def upload_images(sequence_of_images, root='', mirror=False):
    """
    Uploads, NORMALIZE [0,1] and flips horizzontaly (if required)
    an image
    """
    original = list()
    for filename in sequence_of_images:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename) 
            image = ndimage.imread(filepath, mode="RGB") 
            image = image/ 255. # Normalization happens here!
            if not mirror:
                original.append(image)
            else:
                original.append(flip_horizontally(image))
    return np.array(original)
```

## Please note that normalization happens when uploading the images by dividing the pixels' values by 255.


```python
def schedule(track_logs, subsample=1):
    """
    Given a recording's name, the function prepares the response variable (steering angle)
    and the list of image files (only center ones)
    """
    train_share, schedule = list(), list()
    y_train1, y_train2 = list(), list()
    for n, track  in enumerate(track_logs):
        number = track.split('.')[0][-1]
        upload = pd.read_csv(track_logs[n], header = None, names = ['center_img','left_img','right_img','steering_angle','throttle','break','speed'])
        leave_apart = int(len(upload)*0.05)
        upload = upload[leave_apart:-leave_apart]
        target1  = upload.steering_angle.values
        target2 = upload.throttle.values
        upload = upload.center_img.values
        train_pos = np.array([j for j in range(len(upload)) if j % subsample == 0])
        train_share  += list(map(lambda x: x.replace('IMG',track.split('.')[0]), upload[train_pos].tolist()))
        y_train1 += target1[train_pos].tolist()
        y_train2 += target2[train_pos].tolist()
    y_train = np.array(y_train1)
    return train_share, y_train    
```

## Why a generator?

A generator allows to pick up images directly from disk and do not be constrained by the available memory.
In fact, images are loaded as arrays of integers but when normalized they become arrays of float32, occupying more than 
double the initial memory. Running small batches allows to safely perform normalization and other manipulation on the
images without risk of getting out of memory errors. Morevoer, in order to generalize from data, you really need a lot of
examples, more than I actually used in this project. Working with a generator pulling data from disk allows learning
from really huge quantities of data and to build well performing deep learning neural networks.


```python
class data_generator:
    """
    This class is a generator for the neural network based the schedule list of images
    It generates both the original images and (on request) their mirrored image
    """

    def __init__(self, ordering ='th', occlusion=60):
        self.MEAN = 0.0
        self.STD  = 1.0
        self.ROTATE = True
        self.occlusion = occlusion
    
    def fit(self, X):
        self.MEAN = np.mean(X, dtype=np.float32)
        self.STD = np.std(X, dtype=np.float32)
    
    def transform(self, X):
        X = (X - self.MEAN) / self.STD
        
    def cap(self, X, a=0.1, b=0.5):
        X[np.abs(X)<a] = 0.0
        X[X>b] = b
        X[X<(b*-1.0)] = (b*-1.0)
        return X
    
    def moving_average(self, X, n=5) :
        ret = np.cumsum(X, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return np.array(list(X[:(n-1)]) + list(ret[n - 1:] / n))
    
    def flow(self, X, y, batch_size, mirror=False, upper_mask=True):
        repro = int(np.sum(y == 0) / float(np.sum(y > 0)))
        while 1==1:
            for pos in range(0, len(X), batch_size):
                SC = np.array(X[pos:(pos + batch_size)]).copy()
                yb = np.array(y[pos:(pos + batch_size)]).copy()
                Xb = upload_images(SC)
                if upper_mask:
                    Xb[:,:self.occlusion,:,:] = np.zeros((Xb.shape[0],self.occlusion,320,3))
                if self.ROTATE:
                    Xb = rotate(Xb)
                yield Xb, yb
            if mirror:
                for pos in range(0, len(X), batch_size):
                    SC = np.array(X[pos:(pos + batch_size)]).copy()
                    yb = np.array(y[pos:(pos + batch_size)]).copy()*-1.0
                    Xb = upload_images(SC, mirror=True)
                    if upper_mask:
                        Xb[:,:self.occlusion,:,:] = np.zeros((Xb.shape[0],self.occlusion,320,3))
                    if self.ROTATE:
                        Xb = rotate(Xb)
                    yield Xb, yb
```


```python
# The generator is iniatialized
datagen = data_generator()
```

## General discussion about the approach taken

First of all, I've immediately replicated the neural architecture derived from the paper "End to End Learning for Self Driving Cars". Since the very beginning I've taken care to insert dropout and ReLu activations. I also inserted batchnormalization when finding out its properties after reading about it in the forums. Batchnormalization allowed me to pass from Adam to SGD optimization which provided me better and faster results. As for as the number of epochs, I've decided to use early stop on validation loss in order not to train too much the network and have it overfit to the data.

As prescribed by the problem I've prepared various recording of laps with and without recovery on track 1. I set the generator in order to provide me both training and validation data. As for as test data, I thought recording a few laps on track 2.

At the beginning the network just output bad steering. Therefore, confident that the architecture, optimization and early stop were the right choices (I thought anyway that the architecture was a bit overkill, so I tested different dropout values, but in the end 0.2 was always the best). Consequently I concentrated on data. I tried both smoothing (moving average) and capping the recorded steering data. Capping the high end proved the right decision. I also masked the upper part of the image in order to avoid the network to learn about the scenery. Shuffling the examples was the nexr thing that I put into the pipeline.

In spite of generally low errors on both training and validation set (from 0.05 to 0.02 MSE), when the network was tested it keep on giving bad results. The break throught arrived when I decided to strongly subsample the examples with zero steering and oversample the right angles. By doing so the car finally managed its first complete lap, though in a clearly "drunken" mode.

A a final step, I therefore decided to add a few more zero steering examples, providing stability to the car. I capped the training steering angles to 0.6 (lower led the car to go straight on curves, higher made the car shaky again).

Examining the training, validation and test results, split by angle (left, center, right), I noticed that the model didn't fit too well the recovery recordings. Most likely, in my opinion, if I want to improve more the model, I will have to introduce better recovery registrations.


```python
# The schedule for training and validation is built on the basis of multiple recordings, some clockwise, some counter
# clockwise, some just showing recovery. The recording are all based on track 1
# Later in the code I will use also track 2 for testing
train_schedule = ['track_a_b.csv', 'lap4_recovery.csv', 'track_a_f.csv', 'recovery.csv', 
                  'recovery3.csv', 'lap1.csv', 'track1.csv', 'track2.csv', 'track4.csv']
validation_schedule = ['track3.csv']
X_train_sc, y_train = schedule(train_schedule)
X_valid_sc, y_valid = schedule(validation_schedule)
```


```python
# Printing how the upper part of the imaged is blackened in order to limit the 
# details for the neural network to the road, not the scenery
% matplotlib inline
import matplotlib.pyplot as plt
image = ndimage.imread(X_train_sc[0], mode="RGB") / 255.
image[:60,:,:] = np.zeros((60,320,3))
plt.imshow(image)
plt.show()
image.shape
```


![png](output_16_0.png)





    (160L, 320L, 3L)



## Using masking, there is no way the neural network can learn the scenery. It will learn just the road.


```python
# Plotting the stearing angle of the training data (1500 points)
% matplotlib inline
draw = pd.Series(y_train[1000:2500]).plot()
plt.show()
```


![png](output_18_0.png)



```python
# The training data is prepared in order for the neural network to learn
# to drive on the road
# First, the data is capped to 0.6 (no high steering angles)
smooth = 1
digits = 5
X_sc = np.array(X_train_sc)
y_sc = datagen.cap(np.array(y_train), 0.00, 0.60)
y_sc = np.round(datagen.moving_average(y_sc, smooth), digits)
```


```python
# Plotting the capped stearing angle of the training data (1500 points)
% matplotlib inline
draw = pd.Series(y_sc[1000:2500]).plot()
plt.show()
```


![png](output_20_0.png)



```python
# In order to reinforce certain behaviours in the car (drive straight, recover from sides, turn right as lef)
# the cases are choosen on the basis of the type (ordinary or recovery), oversampling right angles and steering
# in general and undersampling straight driving
reinforce_correction = [n for n,(x,y) in enumerate(zip(X_sc, y_sc)) if 'recovery' in x and np.abs(y)>0.05]
reinforce_steering   = [n for n,(x,y) in enumerate(zip(X_sc, y_sc)) if np.abs(y)>0.05]
reinforce_right_steering  = [n for n,(x,y) in enumerate(zip(X_sc, y_sc)) if y>0.05]
reinforce_straight_steering  = [m for p,m in enumerate([n for n,(x,y) in enumerate(zip(X_sc, y_sc)) if y==0]) if p%15==0]
reinforce = np.array(reinforce_correction + reinforce_steering + reinforce_right_steering + reinforce_right_steering + reinforce_straight_steering)
X_sc = X_sc[reinforce]
y_sc = y_sc[reinforce]
```


```python
# The training data distribution is printed, pointing out the mean stearing angle
print ("Steering distribution for training -> mean:%0.3f count:[%i|%i|%i]" % (np.mean(y_sc), np.sum(y_sc<0), np.sum(y_sc==0.0), np.sum(y_sc>0)))
```

    Steering distribution for training -> mean:0.003 count:[10593|3344|10899]
    


```python
# The training data is shuffled in order to present the neural network with varied examples
# Moreover, since the optimization algorithm is SGD, it naturally expects randomly picked
# examples in order to work better
shuffle = True
selection = np.array(list(range(len(y_sc))))
if shuffle:
    np.random.shuffle(selection) 
X_train = list(X_sc[selection])
y_train = y_sc[selection]

examples = len(y_train)
```


```python
# The newly generated distribution of steering angles is now presented
% matplotlib inline
draw = pd.Series(y_train[1000:2500]).plot()
plt.show()
```


![png](output_24_0.png)



```python
# The number of epoch is set high because we will use an early stop method
nb_epoch = 20

# This batch size is suitable for running on my GPU (larger batches will cause out-of-memory)
batch = 128

# Here we can decide if to use the mirroring of images (in order to increase their number 
# and balance the number of angle directions) and masking the upper part of the image
mirror = True
mask   = True
if mirror:
    samples = examples*2
else:
    samples = examples

# Early stopping is set on validation loss, with a patience of 3 epochs
callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1)]

# The model is fit using the generator
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch, mirror=mirror, upper_mask=mask),
                    samples_per_epoch=samples, 
                    nb_epoch=nb_epoch, verbose=1,
                    validation_data=datagen.flow(X_valid_sc, y_valid, batch_size=batch, upper_mask=mask),
                    nb_val_samples=len(X_valid_sc),
                    callbacks=callbacks)
```

    Epoch 1/20
    49672/49672 [==============================] - 375s - loss: 0.1168 - mean_squared_error: 0.1168 - val_loss: 0.0626 - val_mean_squared_error: 0.0626
    Epoch 2/20
    49672/49672 [==============================] - 318s - loss: 0.0614 - mean_squared_error: 0.0614 - val_loss: 0.0633 - val_mean_squared_error: 0.0633
    Epoch 3/20
    49672/49672 [==============================] - 245s - loss: 0.0567 - mean_squared_error: 0.0567 - val_loss: 0.0632 - val_mean_squared_error: 0.0632
    Epoch 4/20
    49672/49672 [==============================] - 299s - loss: 0.0544 - mean_squared_error: 0.0544 - val_loss: 0.0607 - val_mean_squared_error: 0.0607
    Epoch 5/20
    49672/49672 [==============================] - 339s - loss: 0.0521 - mean_squared_error: 0.0521 - val_loss: 0.0596 - val_mean_squared_error: 0.0596
    Epoch 6/20
    49672/49672 [==============================] - 337s - loss: 0.0507 - mean_squared_error: 0.0507 - val_loss: 0.0627 - val_mean_squared_error: 0.0627
    Epoch 7/20
    49672/49672 [==============================] - 336s - loss: 0.0497 - mean_squared_error: 0.0497 - val_loss: 0.0570 - val_mean_squared_error: 0.0570
    Epoch 8/20
    49672/49672 [==============================] - 351s - loss: 0.0487 - mean_squared_error: 0.0487 - val_loss: 0.0599 - val_mean_squared_error: 0.0599
    Epoch 9/20
    49672/49672 [==============================] - 349s - loss: 0.0474 - mean_squared_error: 0.0474 - val_loss: 0.0607 - val_mean_squared_error: 0.0607
    Epoch 10/20
    49672/49672 [==============================] - 339s - loss: 0.0465 - mean_squared_error: 0.0465 - val_loss: 0.0583 - val_mean_squared_error: 0.0583
    Epoch 11/20
    49672/49672 [==============================] - 364s - loss: 0.0461 - mean_squared_error: 0.0461 - val_loss: 0.0642 - val_mean_squared_error: 0.0642
    Epoch 00010: early stopping
    


```python
# The learnig is visualized. The final epoch is the 10th 
batches = range(1,len(history.history['mean_squared_error'])+1)
loss_plot = plt.subplot(211)
loss_plot.set_title('Train')
loss_plot.plot(batches, history.history['mean_squared_error'], 'g')
loss_plot.axes.get_xaxis().set_ticks([])
acc_plot = plt.subplot(212)
acc_plot.set_title('Validation')
acc_plot.plot(batches, history.history['val_mean_squared_error'], 'r')
acc_plot.set_xlim([batches[0], batches[-1]])
```




    (1, 11)




![png](output_26_1.png)



```python
# Now that the neural network is built, I test it on the train, validation and test data in order to
# highlight what is working and what is not (is it good predicting steering left or right? What about going straight?)

# The test is done on track 2, therefore this is a test of generalization (it actually cannot run on track 2...)

from sklearn.metrics import mean_squared_error
train_schedule = ['track_a_b.csv', 'lap4_recovery.csv', 'track_a_f.csv', 'recovery.csv', 'recovery2.csv', 'recovery3.csv', 'lap1.csv']
validation_schedule = ['track3.csv']
test_schedule = ['track_b_fb.csv']
check_schedule =[('TRAIN', train_schedule),('VALIDATION', validation_schedule),('TEST', test_schedule)]

for label, a_schedule in check_schedule:
    print ('\n'+label+'\n'+'-'*len(label))
    for j in range(len(a_schedule)):
        X_train_sc, y_train = schedule([a_schedule[j]])
        error = 0.0
        y_true, y_pred = list(),list()
        for n,(img, response) in enumerate(zip(X_train_sc, y_train)):
            image = ndimage.imread(img, mode="RGB") 
            image = image/ 255. # Normalization
            image[:70,:,:] = np.zeros((70,320,3))
            image = np.swapaxes(np.swapaxes(image,1,2),0,1)
            transformed_image_array = image[None, :, :, :]
            steering_angle = float(model.predict(transformed_image_array, batch_size=1))
            error += (response - steering_angle)**2
            MSE = (error / float(n+1))
            y_true.append(response)
            y_pred.append(steering_angle)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        MSE_straight = mean_squared_error(y_true=y_true[y_true == 0.0], y_pred=y_pred[y_true==0])
        MSE_right = mean_squared_error(y_true=y_true[y_true > 0.0], y_pred=y_pred[y_true > 0])
        MSE_left = mean_squared_error(y_true=y_true[y_true < 0.0], y_pred=y_pred[y_true < 0])

        print ("%s (mean=%0.3f, strength:%0.3f, var:%0.3f): MSE_tot:%0.3f MSE_lsr:[%0.3f|%0.3f|%0.3f]" % (a_schedule[j], np.mean(y_true), 
                                                                                                  np.mean(np.abs(y_true)>0.5), np.var(y_true), 
                                                                                                  MSE, MSE_left,MSE_straight, MSE_right))
```

    
    TRAIN
    -----
    track_a_b.csv (mean=0.039, strength:0.036, var:0.033): MSE_tot:0.034 MSE_lsr:[0.044|0.025|0.058]
    lap4_recovery.csv (mean=-0.041, strength:0.027, var:0.029): MSE_tot:0.037 MSE_lsr:[0.043|0.034|0.053]
    track_a_f.csv (mean=-0.041, strength:0.028, var:0.027): MSE_tot:0.040 MSE_lsr:[0.042|0.038|0.060]
    recovery.csv (mean=-0.047, strength:0.138, var:0.117): MSE_tot:0.095 MSE_lsr:[0.161|0.065|0.135]
    recovery2.csv (mean=-0.042, strength:0.030, var:0.032): MSE_tot:0.040 MSE_lsr:[0.025|0.025|0.189]
    recovery3.csv (mean=-0.070, strength:0.095, var:0.069): MSE_tot:0.098 MSE_lsr:[0.205|0.060|0.202]
    lap1.csv (mean=-0.040, strength:0.031, var:0.030): MSE_tot:0.043 MSE_lsr:[0.052|0.039|0.080]
    
    VALIDATION
    ----------
    track3.csv (mean=-0.041, strength:0.087, var:0.051): MSE_tot:0.065 MSE_lsr:[0.150|0.044|0.212]
    
    TEST
    ----
    track_b_fb.csv (mean=0.013, strength:0.102, var:0.082): MSE_tot:0.118 MSE_lsr:[0.253|0.030|0.166]
    


```python
# The model's network architecture is saved to disk
json_string = model.to_json()
with open('model.json', 'w') as json_file:
    json.dump(json_string, json_file)
```


```python
# The model's weights are saved to disk
model.save_weights('model.h5')
```
