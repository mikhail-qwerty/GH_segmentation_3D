import numpy as np
import tensorflow as tf
import random
import glob
import os 

import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, concatenate
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow import Tensor

from tensorflow.keras.layers import Conv3D, UpSampling3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.layers import Conv3DTranspose

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# set visible GPU devices 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# define the minimal compute copability. 
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="6" 

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[:], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def dataGenerator(train_dir, image_dir, mask_dir, batch_size, target_size = (128, 128, 128)):  
    i = 0 
    im_list = glob.glob(train_dir + '/' + image_dir + '/' + '*.tif')
    mask_list = glob.glob(train_dir + '/' + mask_dir + '/' + '*.tif')
    while True:
        image_batch = []
        mask_batch = []
        for b in range(batch_size):
          
            if i == len(im_list):
                i = 0
                data_list = list(zip(im_list, mask_list))
                random.shuffle(data_list)
                im_list, mask_list = zip(*data_list)

            sample_im = im_list[i]
            sample_mask = mask_list[i]
            i += 1
            im = io.imread(sample_im, as_gray=True)
            mask = io.imread(sample_mask, as_gray=True)

            im = trans.resize(im,target_size)
            mask = trans.resize(mask,target_size)

            im = np.reshape(im,im.shape + (1,))
            mask = np.reshape(mask, mask.shape + (1,))
            
            image_batch.append(im)
            mask_batch.append(mask)
            
        yield (np.array(image_batch), np.array(mask_batch))

def UNet_3D(input_size = (128, 128, 128, 1)):

    inputs = tf.keras.Input(input_size)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(drop5), conv4], axis=4)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv3], axis=4)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

    model = Model(inputs = inputs, outputs= conv10)

    return model
    
model = UNet_3D(input_size=(128, 128, 128,1))

batch_size = 1
steps_per_epoch = 378//batch_size

myGene = dataGenerator(train_dir = 'train/256', image_dir = 'images', mask_dir = 'labels', batch_size = batch_size)
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
model_checkpoint = ModelCheckpoint('unet_3D.hdf5', monitor='loss',verbose=1, save_best_only=True)

history = model.fit(myGene,steps_per_epoch=steps_per_epoch,epochs=100,callbacks=[model_checkpoint])