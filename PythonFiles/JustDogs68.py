
# coding: utf-8

# In[1]:


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
from PIL import Image
import skimage

from keras.models import model_from_json


# In[2]:


trainFile="Desktop/vedem"
testFile='Desktop/Dataset/Test_Dogs/'
validationFile='Desktop//Dataset/Validation_Dogs/'


# In[3]:


base_model=MobileNet(weights='imagenet',include_top=False)
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) 
x=Dense(512,activation='relu')(x) 
output=Dense(127,activation='softmax')(x) 


# In[4]:


model=Model(inputs=base_model.input,outputs=output)
for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True


# In[6]:


len(model.layers)


# In[92]:


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,horizontal_flip=True,rotation_range=10)
train_generator=train_datagen.flow_from_directory(trainFile,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=1,
                                                 class_mode='categorical',shuffle=False,subset='training')


# In[6]:


validation_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = validation_datagen.flow_from_directory(
    validationFile,
     target_size=(224,224),
    color_mode='rgb',
    batch_size=50,
    class_mode='categorical',shuffle=False)


# In[8]:


test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    testFile,
     target_size=(224,224),
    color_mode='rgb',
    batch_size=50,
    class_mode='categorical',shuffle=True)


# In[9]:


filepath="/home/ficiu/Desktop/keras/JustDogsMobileNet{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[10]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size

model.fit_generator(generator=train_generator,steps_per_epoch=step_size_train, epochs=100
                    , validation_data = test_generator
                    , validation_steps = test_generator.samples // step_size_train
                    , verbose=1
                    , callbacks=callbacks_list)


# In[53]:


import os
os.remove("Desktop/Dataset/Train_cat_dog/Cat/1CatFara rasa17624.jpg")


# In[11]:



plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Percent')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[12]:


plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model accuracy')
plt.ylabel('Percent')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[13]:


model.history.history


# In[201]:


pic=Image.open('/home/ficiu/Desktop/vedem/Dog/affenpinscher109.jpg')


# In[202]:


pic


# In[183]:


train=model.predict_generator(validation_generator,steps=2,verbose=1)


# In[184]:


np.argmax(train,axis=1)


# In[191]:


validation_generator.classes


# In[198]:


a=zip(validation_generator.filenames,validation_generator.classes)
b=zip(a,np.argmax(train,axis=1))


# In[199]:


for each in b:
    print (each)


# In[33]:


a=train_generator.next()


# In[44]:


a[0][0].shape


# In[38]:


from skimage import data
from skimage.viewer import ImageViewer

viewer = ImageViewer(b[0][0])
viewer.show()


# In[80]:


b=train_generator.next()


# In[89]:


b[0][0]


# In[93]:


for i in range(50):
    a=train_generator.next()
    skimage.io.imsave('/home/ficiu/Desktop/Raport/Exemple poze augmentate/asdas'+str(i)+'.jpg',a[0][0])


# In[ ]:


norm_image = cv2.normalize(b[0][0], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

norm_image.astype(np.uint8)


# In[ ]:


skimage.io.imshow(norm_image)

