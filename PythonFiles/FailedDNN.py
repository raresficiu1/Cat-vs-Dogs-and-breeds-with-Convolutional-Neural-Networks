
# coding: utf-8

# Incarc datele si le transform in vectori de 1

# In[2]:


import glob
import skimage
import os
from skimage import io
import matplotlib
import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected
from skimage.transform import rescale, resize
import random


# In[3]:


cd Desktop


# In[4]:


cd Train_data/


# In[5]:


dogs_races = glob.glob('/home/ficiu/Desktop/Train_data/Caini/Cu rasa/*')


# In[6]:


cats_races=glob.glob('/home/ficiu/Desktop/Train_data/Pisici/Cu rasa/*')


# In[7]:


X_cats_dogs=[]
y_cats_dogs=[]

X_races=[]
y_races=[]
z_races=[]


# In[8]:


for each in dogs_races:
    for i in glob.glob(each+'/*'):
        X_races.append(i)
        y_races.append(each.split('/')[-1])
        z_races.append(1)


# In[9]:


for each in cats_races:
    for i in glob.glob(each+'/*'):
        X_races.append(i)
        y_races.append(each.split('/')[-1])
        z_races.append(0)


# In[10]:


dogs_without_race=glob.glob('/home/ficiu/Desktop/Train_data/Caini/Fara rasa/*')


# In[11]:


cats_without_race=glob.glob('/home/ficiu/Desktop/Train_data/Pisici/Fara rasa/*')


# In[12]:


for each in dogs_races:
    for i in glob.glob(each+'/*'):
        X_cats_dogs.append(i)
        y_cats_dogs.append('1')
        
        
for each in cats_races:
    for i in glob.glob(each+'/*'):
        X_cats_dogs.append(i)
        y_cats_dogs.append('0')
for each in dogs_without_race:
    X_cats_dogs.append(i)
    y_cats_dogs.append('1')

for each in cats_without_race:
    X_cats_dogs.append(i)
    y_cats_dogs.append('0')


# In[13]:


len(X_races),len(y_races),len(z_races)


# In[14]:


len(X_cats_dogs)


# In[15]:


random.seed(125)
random.shuffle(X_cats_dogs)
random.shuffle(y_cats_dogs)
# X_cats_dogs_validation=X_cats_dogs[0:20]
# y_cats_dogs_validation=y_cats_dogs[0:20]
# X_cats_dogs_test=X_cats_dogs[0:100]
# y_cats_dogs_test=y_cats_dogs[0:100]
# X_cats_dogs=X_cats_dogs[100:500]
# y_cats_dogs=y_cats_dogs[100:500]


# imagenet e pe 224x224

# In[16]:


pic_width=100
pic_height=100
n_inputs = pic_width*pic_height*3
n_hidden1= 300
n_hidden2= 50
n_hidden3= 30
n_outputs = 2


# In[17]:


X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int64,shape=(None),name="y")


# In[18]:


with tf.name_scope("really_deep_nn"):
    hidden1=fully_connected(X,n_hidden1,scope='hidden1')
    hidden2=fully_connected(hidden1,n_hidden2,scope='hidden2')
    hidden3=fully_connected(hidden2,n_hidden3,scope='hidden3')
    output=fully_connected(hidden3,n_outputs,scope="outputs",activation_fn=None)


# In[19]:


with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=output)
    loss=tf.reduce_mean(xentropy,name="loss")


# In[20]:


learning_rate=0.001
with tf.name_scope("train"):
    optimizer=tf.train.AdamOptimizer(learning_rate)
    training_op=optimizer.minimize(loss)


# In[21]:


with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(output,y,1)
    accuracy =tf.reduce_mean(tf.cast(correct,tf.float32))


# In[22]:


init =tf.global_variables_initializer()
saver=tf.train.Saver()


# In[ ]:


x,y=prepare_batch(300,0,X_cats_dogs,y_cats_dogs,n_)


# In[33]:


for i in x:
    print (i)


# In[24]:


def prepare_batch(batch_size ,start, X_cats_dogs, y_cats_dogs, n_inputs):
    x_batch=np.zeros(batch_size*n_inputs).reshape(batch_size, n_inputs) #matricea de input
    y_batch=np.zeros(batch_size)   #matricea label
    
    for i in range(batch_size):
        poza_resized = resize(io.imread(X_cats_dogs[start:start+batch_size][i]),(pic_height,pic_width),anti_aliasing=True)
        
        x_batch[i]=poza_resized.flatten()
        y_batch[i]=y_cats_dogs[start:start+batch_size][i]

    return x_batch , y_batch


# In[23]:


#x,y = prepare_batch(50,120,X_cats_dogs,y_cats_dogs,400*400*3)


# In[24]:


n_epochs=10
batch_size=150

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# In[26]:


with tf.Session(config = config) as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(10):
            X_batch, y_batch=prepare_batch(batch_size,batch_size*iteration,X_cats_dogs,y_cats_dogs,n_inputs)
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
            train_accuracy=accuracy.eval(feed_dict={X:X_batch,y:y_batch})
            print (iteration," Train accuracy " ,train_accuracy)
        test_accuracy=0
        
        for i in range(10):
            X_batch_test,y_batch_test = prepare_batch(batch_size,batch_size*i+10,X_cats_dogs,y_cats_dogs,n_inputs) 
            test_accuracy+=accuracy.eval(feed_dict={X:X_batch_test,y:y_batch_test})
        print("Accuracy:" ,test_accuracy/10)
    save_path=saver.save(sess,"./my_model_final.ckpt")

