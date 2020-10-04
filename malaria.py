#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:55:58 2020

@author: surya.ketaraju
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:30:39 2020

@author: surya.ketaraju
"""

import pandas as pd
import numpy as np
from matplotlib import image
import keras
from keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(rescale=1./255.,
                           shear_range = 0.2,
                           zoom_range = 0.2,
                           horizontal_flip = True,
                           validation_split=0.25)

test_datagen = ImageDataGenerator(rescale=1./255.)


directory='/home/FRACTAL/surya.ketaraju/tensorflow_datasets/downloads/extracted/ZIP.ceb.nlm.nih.gov_proj_malaria_cell_imagesCLJ1vGxXDKcJmHnYfoar_K3ipRQWtxIVA-imvIbvBbs.zip/cell_images'
test_directory='/home/FRACTAL/surya.ketaraju/tensorflow_datasets/downloads/extracted/ZIP.ceb.nlm.nih.gov_proj_malaria_cell_imagesCLJ1vGxXDKcJmHnYfoar_K3ipRQWtxIVA-imvIbvBbs.zip'

train_set = datagen.flow_from_directory(
                                        directory,
                                        target_size=(64,64),
                                        
                                        color_mode='rgb',
                                        class_mode='binary',
                                        subset='training',
                                        shuffle=True,
                                        )

validation_set = datagen.flow_from_directory(
                                        directory,
                                        target_size=(64,64),
                                      
                                        color_mode='rgb',
                                        class_mode='binary',
                                        subset='validation',
                                        shuffle=True
                                        )

test_generator= test_datagen.flow_from_directory(
                                        test_directory,
                                        target_size=(64,64),
                                        classes=['test'],
                                        
                                        
                                        color_mode='rgb',                                        
                                        
                                        )

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input


model = Sequential()
model.add(Conv2D(32, (5,5),padding = 'Same', activation ='tanh', input_shape = (64,64,3)))
model.add(Conv2D(32, (5,5),padding = 'Same', activation ='tanh'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128, activation = "tanh"))
model.add(Dropout(0.1))
model.add(Dense(1, activation = "sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
### CNN using Transfer learning with VGG16 
#base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3))
#
#add_model = Sequential()
#add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
#add_model.add(Dense(256, activation='relu'))
#add_model.add(Dense(1, activation='softmax'))
#
#model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
#
#model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])
#


model.fit_generator(train_set,
                    steps_per_epoch = 12000,
                    epochs = 10,
                    validation_data = validation_set,
                    validation_steps = 55)
                            
model.evaluate_generator(generator=validation_set, steps=31)



pred=model.predict_generator(test_generator,
steps=5,
verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_set.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


filenames=test_generator.filenames
results=pd.DataFrame({"Image":filenames,
                      "target":predictions})
results.to_csv("dance_results.csv",index=False)


