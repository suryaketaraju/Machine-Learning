#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 23:16:26 2021

@author: surya.ketaraju
"""

#IMPORTING LIBRARIES

import tensorflow as tf
import os
import random
import numpy as np
import cv2
 
from tqdm import tqdm 

from skimage import io
from skimage import transform
import matplotlib.pyplot as plt


#GATHERING AND PREPROCESSING DATA

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

#add paths
TRAIN_PATH = '/home/FRACTAL/surya.ketaraju/Downloads/image_seg/data/'
TEST_PATH = '/home/FRACTAL/surya.ketaraju/Downloads/image_seg/test/'

#get the names of the image files
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[2]

#Load X and Y with zeroes to fill with data later
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)


#the filenames are stored in train_ids/test_ids so we will join the path and filenames to get to the images
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_
    img = io.imread(path + '/image/' + id_ + '.jpg')[:,:,:IMG_CHANNELS]  
    img = transform.resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=bool)
    #fill X with the img value
    X_train[n] = img  
    #gathering the mask files
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/mask/'))[2]:
        mask_new = io.imread(path + '/mask/' + mask_file)
        #reducing the channels
        mask_new = cv2.cvtColor(mask_new, cv2.COLOR_BGR2GRAY)
        #adding a dimension since the above command dropped all channels
        mask_new = np.expand_dims(transform.resize(mask_new, (IMG_HEIGHT, IMG_WIDTH), mode='reflect', preserve_range=bool), axis=-1)
        #masking
        mask = np.maximum(mask_new, mask)

    Y_train[n] = mask #load Y train with the mask data
    

#same process for test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH
    img = io.imread(path + id_)[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = transform.resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img


X_train = X_train/255
Y_train = Y_train/255
X_test = X_test/255

#DISPLAYING THE IMAGE AND THE CORRESPONDING MASK
    
img_new = random.randint(0, len(train_ids))
io.imshow(X_train[img_new])
plt.show()
io.imshow(np.squeeze(Y_train[img_new]))
plt.show()


#IMPORTING KERAS UNET MODEL
from keras_unet.models import custom_unet

model = custom_unet(
    input_shape=(128, 128, 3),
    use_batch_norm=False,
    num_classes=1,
    filters=128,
    dropout=0.2,
    output_activation='sigmoid')

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

results = model.fit(X_train, Y_train, validation_split=0.1, epochs=20)

from keras_unet.utils import plot_segm_history

plot_segm_history(results, metrics = ['accuracy','val_accuracy'], losses=['loss', 'val_loss']) 



#### MAKING PREDICTIONS ###

preds_train = model.predict(X_train)
preds_val = model.predict(X_train)
preds_test = model.predict(X_test)

 
preds_train_t = (preds_train > 0.45)
preds_val_t = (preds_val > 0.45)
preds_test_t = (preds_test > 0.45)


#VISUALIZING THE MASKS
i = random.randint(0, len(preds_train_t))
io.imshow(X_train[i])
plt.show()

io.imshow(np.squeeze(Y_train[i]))
plt.show()
 
io.imshow(np.squeeze(preds_train_t[i]))
plt.show()
