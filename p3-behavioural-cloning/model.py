
# coding: utf-8

# ## Luca Massaron: Model for Behavioural Cloning based on the paper 
# "End to End Learning for Self-Driving Cars" by NVIDIA engineers.

# In[1]:

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


# In[3]:

# I set the Keras backend to Theano, in fact this model has been built on a Windows machine, running Theano since working on 
# a Linux virtual machine proved impossible. The model can run on both Theano and Tensorflow backend, both on Windows or Linux
# just by some slight modification to the structure of images
from keras import backend as K
K.set_image_dim_ordering('th')


# In[5]:

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


# In[6]:

# The input shape of the images is defined and feed into the model which is compiled
# Please notice that we are working with Theano, so the color channels are put before the width and height
# This implies that we have to modify any image feed into the net accordingly to this schema
color_type_global = 3
img_rows, img_cols = 160, 320
model = ETE(None, color_type_global, img_rows, img_cols)


# In[7]:

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
    Uploads, normalize and flips horizzontaly (if required)
    an image
    """
    original = list()
    for filename in sequence_of_images:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename) 
            image = ndimage.imread(filepath, mode="RGB") 
            image = image/ 255. # Normalization
            if not mirror:
                original.append(image)
            else:
                original.append(flip_horizontally(image))
    return np.array(original)


# In[8]:

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


# In[9]:

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


# In[11]:

# The generator is iniatialized
datagen = data_generator()


# In[12]:

# The schedule for training and validation is built on the basis of multiple recordings, some clockwise, some counter
# clockwise, some just showing recovery. The recording are all based on track 1
# Later in the code I will use also track 2 for testing
train_schedule = ['track_a_b.csv', 'lap4_recovery.csv', 'track_a_f.csv', 'recovery.csv', 
                  'recovery3.csv', 'lap1.csv', 'track1.csv', 'track2.csv', 'track4.csv']
validation_schedule = ['track3.csv']
X_train_sc, y_train = schedule(train_schedule)
X_valid_sc, y_valid = schedule(validation_schedule)


# In[13]:

# Printing how the upper part of the imaged is blackened in order to limit the 
# details for the neural network to the road, not the scenery
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
image = ndimage.imread(X_train_sc[0], mode="RGB") / 255.
image[:60,:,:] = np.zeros((60,320,3))
plt.imshow(image)
plt.show()
image.shape


# In[15]:

# Plotting the stearing angle of the training data (1500 points)
get_ipython().magic(u'matplotlib inline')
draw = pd.Series(y_train[1000:2500]).plot()
plt.show()


# In[16]:

# The training data is prepared in order for the neural network to learn
# to drive on the road
# First, the data is capped to 0.6 (no high steering angles)
smooth = 1
digits = 5
X_sc = np.array(X_train_sc)
y_sc = datagen.cap(np.array(y_train), 0.00, 0.60)
y_sc = np.round(datagen.moving_average(y_sc, smooth), digits)


# In[17]:

# Plotting the capped stearing angle of the training data (1500 points)
get_ipython().magic(u'matplotlib inline')
draw = pd.Series(y_sc[1000:2500]).plot()
plt.show()


# In[18]:

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


# In[20]:

# The training data distribution is printed, pointing out the mean stearing angle
print ("Steering distribution for training -> mean:%0.3f count:[%i|%i|%i]" % (np.mean(y_sc), np.sum(y_sc<0), np.sum(y_sc==0.0), np.sum(y_sc>0)))


# In[21]:

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


# In[22]:

# The newly generated distribution of steering angles is now presented
get_ipython().magic(u'matplotlib inline')
draw = pd.Series(y_train[1000:2500]).plot()
plt.show()


# In[24]:

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


# In[25]:

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


# In[30]:

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


# In[27]:

# The model's network architecture is saved to disk
json_string = model.to_json()
with open('model.json', 'w') as json_file:
    json.dump(json_string, json_file)


# In[28]:

# The model's weights are saved to disk
model.save_weights('model.h5')

