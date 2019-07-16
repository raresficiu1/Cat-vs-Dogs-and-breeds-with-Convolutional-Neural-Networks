
# coding: utf-8

# In[2]:


import tensorflow.keras
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint


# In[3]:


trainFile="Desktop/Dataset/Train_cat_dog/"
testFile='Desktop/Dataset/Test_cat_dog/'
validationFile='Desktop/Dataset/Validation_cat_dog/'


# In[4]:


base_model=MobileNet(weights='imagenet',include_top=False)

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
preds=Dense(2,activation='softmax')(x) 


# In[7]:


model=Model(inputs=base_model.input,outputs=preds)
for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True


# In[8]:


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=10,horizontal_flip=True)

train_generator=train_datagen.flow_from_directory(trainFile,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=50,
                                                 class_mode='categorical',shuffle=True,subset='training')


# In[9]:


validation_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = validation_datagen.flow_from_directory(
    validationFile,
     target_size=(224,224),
    color_mode='rgb',
    batch_size=50,
    class_mode='categorical',shuffle=True)


# In[10]:


test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    testFile,
     target_size=(224,224),
    color_mode='rgb',
    batch_size=50,
    class_mode='categorical',shuffle=True)


# In[11]:


filepath="/home/ficiu/Desktop/keras/All{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[32]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size

model.fit_generator(generator=train_generator,steps_per_epoch=step_size_train, epochs=100
                    , validation_data = test_generator
                    , validation_steps = test_generator.samples // step_size_train
                    , verbose=1
                    , callbacks=callbacks_list)


# In[10]:


model.history.history


# In[12]:


plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Percent')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[13]:


plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model accuracy')
plt.ylabel('Percent')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[33]:


model.history.history


# In[34]:


plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Percent')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[36]:


plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[1]:


#65 normal 71 cu augmentat

