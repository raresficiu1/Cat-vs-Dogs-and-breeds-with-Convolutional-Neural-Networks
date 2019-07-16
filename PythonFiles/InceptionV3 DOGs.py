
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
from keras import optimizers


# In[2]:


trainFile="Desktop/Dataset/Train_Dogs/"
testFile='Desktop/Dataset/Test_Dogs/'


# In[4]:


base_model=keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
preds=Dense(127,activation='softmax')(x)


# In[6]:


model=Model(inputs=base_model.input,outputs=preds)


# In[7]:


model.layers


# In[12]:


for layer in model.layers[:130]:
    layer.trainable=False
for layer in model.layers[130:]:
    layer.trainable=True


# In[3]:


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=10,horizontal_flip=True)

train_generator=train_datagen.flow_from_directory(trainFile,
                                                 target_size=(299,299),
                                                 color_mode='rgb',
                                                 batch_size=50,
                                                 class_mode='categorical',shuffle=True,subset='training')


# In[9]:


test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    testFile,
     target_size=(299,299),
    color_mode='rgb',
    batch_size=50,
    class_mode='categorical',shuffle=True)


# In[10]:


filepath="/home/ficiu/Desktop/keras/InceptionV3DOGS{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[16]:


adam=optimizers.adam(lr=0.001,decay=0.01)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size

model.fit_generator(generator=train_generator,steps_per_epoch=step_size_train, epochs=100
                    , validation_data = test_generator
                    , validation_steps = test_generator.samples // step_size_train
                    , verbose=1
                    , callbacks=callbacks_list)


# In[17]:


model.history.history


# In[ ]:


loss: 2.0053 - acc: 0.4339 - val_loss: 2.4701 - val_acc: 0.4133


# In[18]:


plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Percent')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[21]:


plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[16]:


model.history.history


# In[17]:


plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Percent')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[18]:


plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model accuracy')
plt.ylabel('Percent')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

