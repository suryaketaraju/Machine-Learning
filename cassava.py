#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:32:04 2020

@author: surya.ketaraju
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:59:19 2020

@author: surya.ketaraju
"""

"""
Created on Wed Apr 22 11:04:21 2020

@author: surya.ketaraju
"""
import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.applications.resnet50 import ResNet50, preprocess_input

#from keras import regularizers, optimizers
import pandas as pd
import numpy as np
from keras import backend as K 
#K.clear_session()


#initialize the CNN by creating an object
model = Sequential()
model.add(Conv2D(32, (5,5),padding = 'Same', activation ='relu', input_shape = (224,224,3)))
model.add(Conv2D(32, (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(64, (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
#
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(5, activation = "softmax"))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])
model.summary()
#model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
#rmsprop(lr=0.0001, decay=1e-6)

from keras.applications.vgg16 import VGG16
vggmodel = VGG16(weights='imagenet', include_top=True)

for layers in (vggmodel.layers)[:19]:
    print(layers)
    layers.trainable = False

X= vggmodel.layers[-2].output
predictions = Dense(5, activation="softmax")(X)
model_final = Model(input = vggmodel.input, output = predictions)

model_final.compile(loss = "categorical_crossentropy", 
                    optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', 
                             verbose=1, save_best_only=True, save_weights_only=False, 
                             mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, 
                      verbose=1, mode='auto')

model_final.fit_generator(generator= train_generator, steps_per_epoch= 2,
                          epochs= 100, validation_data= valid_generator, 
                          validation_steps=1, callbacks=[checkpoint,early])

model_final.save_weights("vgg16_1.h5")

#transfer learning model

## CNN using Transfer learning with VGG16 
#from keras.applications.vgg16 import VGG16
#from keras import optimizers


base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
add_model = Sequential()
add_model.add(Flatten())
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(5, activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model()(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])


# Training only top layers i.e. the layers which we have added in the end
for layer in base_model.layers:
    layer.trainable = False

# Step 4 - Full connection
#model.add(Dense(units = 120, activation = 'relu'))

#fitting
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np

traindf=pd.read_csv('/home/FRACTAL/surya.ketaraju/Downloads/cassava leaf disease/cassava-leaf-disease-classification/train.csv', dtype=str)
#testdf=pd.read_csv('/home/FRACTAL/surya.ketaraju/Downloads/COMPETITION/dance/dataset/test/test.csv', dtype=str)

datagen=ImageDataGenerator(rescale=1./255.,
                           shear_range = 0.2,
                           zoom_range = 0.2,
                           horizontal_flip = True,
                           validation_split=0.20)

train_generator=datagen.flow_from_dataframe(
                                            dataframe=traindf,
                                            directory="/home/FRACTAL/surya.ketaraju/Downloads/cassava leaf disease/cassava-leaf-disease-classification/train_images",
                                            x_col="image_id",
                                            y_col="label",
                                            subset="training",
                                            batch_size=6,
                                            validate_filenames=True,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="categorical",
                                            target_size=(64,64))

valid_generator=datagen.flow_from_dataframe(
                                            dataframe=traindf,
                                            directory="/home/FRACTAL/surya.ketaraju/Downloads/cassava leaf disease/cassava-leaf-disease-classification/train_images",
                                            x_col="image_id",
                                            y_col="label",
                                            subset="validation",
                                            validate_filenames=False,
                                            batch_size=32,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="categorical",
                                            target_size=(64,64))

test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_directory(
                                                
                                                directory="/home/FRACTAL/surya.ketaraju/Downloads/cassava leaf disease/cassava-leaf-disease-classification/"
                                                
                                                ,batch_size=32,
                                                seed=42,
                                                shuffle=False,
                                                
                                                classes=['test_images'],
                                                target_size=(64,64))

model.fit_generator(train_generator,
                         steps_per_epoch = 22000,
                         epochs =4,
                         validation_data = valid_generator,
                         validation_steps = 10000)

#history = model.fit_generator(get_training_data(),
#                samples_per_epoch=1, nb_epoch=1,nb_val_samples=5,
#                verbose=1,validation_data=get_validation_data())

#Evaluate the model
model.save_weights('cassava.h5')
model.evaluate_generator(generator=valid_generator,
steps=55)

#Predict the output

test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)

from keras.preprocessing import image



img=image.load_img('/home/FRACTAL/surya.ketaraju/Downloads/COMPETITION/dataset/test/odissi/484.jpg', target_size = (64,64))
img_1=img
img = image.img_to_array(img)
img=img/255
import numpy as np
img = np.expand_dims(img, axis = 0)

dance=model.predict(img)
d=np.argmax(dance)