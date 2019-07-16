
# coding: utf-8

# In[1]:


import tensorflow.keras
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint


# In[2]:


trainFile="Desktop/Dataset/Train_All_races/"
testFile='Desktop/Dataset/Test_All_races/'
validationFile='Desktop/Dataset/Validation_all_races/'


# In[3]:


base_model=keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
preds=Dense(139,activation='softmax')(x)


# In[4]:


model=Model(inputs=base_model.input,outputs=preds)


# In[5]:


model.layers


# In[6]:


for layer in model.layers[:100]:
    layer.trainable=False
for layer in model.layers[130:]:
    layer.trainable=True


# In[7]:


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=10,horizontal_flip=True)

train_generator=train_datagen.flow_from_directory(trainFile,
                                                 target_size=(299,299),
                                                 color_mode='rgb',
                                                 batch_size=50,
                                                 class_mode='categorical',shuffle=True,subset='training')


# In[8]:


validation_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = validation_datagen.flow_from_directory(
    validationFile,
     target_size=(299,299),
    color_mode='rgb',
    batch_size=50,
    class_mode='categorical',shuffle=True)


# In[9]:


test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    testFile,
     target_size=(299,299),
    color_mode='rgb',
    batch_size=50,
    class_mode='categorical',shuffle=True)


# In[10]:


filepath="/home/ficiu/Desktop/keras/InceptionV3All{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[11]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size

model.fit_generator(generator=train_generator,steps_per_epoch=step_size_train, epochs=100
                    , validation_data = test_generator
                    , validation_steps = test_generator.samples // step_size_train
                    , verbose=1
                    , callbacks=callbacks_list)


# In[12]:


model.history.history


# In[13]:


plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Percent')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[14]:


plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model accuracy')
plt.ylabel('Percent')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

