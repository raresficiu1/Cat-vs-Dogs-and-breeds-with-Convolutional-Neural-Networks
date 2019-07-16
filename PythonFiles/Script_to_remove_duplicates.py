
# coding: utf-8

# In[64]:


import glob
import skimage
import os
from skimage import io
import matplotlib
import sys


# In[2]:


cd Desktop


# In[3]:


cd Train_data/


# In[4]:


cd Caini


# In[5]:


cd Cu\ rasa


# Loading the images into memory

# In[6]:


dogs_races = glob.glob('/home/ficiu/Desktop/Train_data/Caini/Cu rasa/*')


# In[7]:


dogs_races


# In[8]:


alldogs =[]


# In[9]:


for eachrace in dogs_races:
    alldogs.append(glob.glob(eachrace+'/*'))


# In[10]:


alldogs


# In[16]:


os.path.getsize(alldogs[1][1])


# In[17]:


dogs_withoutrace=glob.glob('/home/ficiu/Desktop/Train_data/Caini/Fara rasa/*')


# Verifying for doubles

# In[49]:


possibleDoubles=[]
possibleDoublesMirror=[]

for i in range(len(alldogs)):
    for j in range(len(alldogs[i])):
        for k in range(j+1,len(alldogs[i])):
            if(os.path.getsize(alldogs[i][j])== os.path.getsize(alldogs[i][k])):
                possibleDoubles.append(alldogs[i][j])
                possibleDoublesMirror.append(alldogs[i][k])
        for u in range(len(dogs_withoutrace)):
            if(os.path.getsize(alldogs[i][j])== os.path.getsize(dogs_withoutrace[u])):
                possibleDoubles.append(alldogs[i][j])
                possibleDoublesMirror.append(dogs_withoutrace[u])
    print(i)


# In[50]:





# In[186]:


i = skimage.io.imread(similarfinalcats[4])
skimage.io.imshow(i)


# In[ ]:


j = skimage.io.imread(similarfinal2cats[4])
skimage.io.imshow(j)


# Doing the same thing for the cats folders

# In[53]:


cats_races=glob.glob('/home/ficiu/Desktop/Train_data/Pisici/Cu rasa/*')
allcats=[]
for eachrace in cats_races:
    allcats.append(glob.glob(eachrace+'/*'))


# In[55]:


cats_withoutrace=glob.glob('/home/ficiu/Desktop/Train_data/Pisici/Fara rasa/*')


# In[56]:


possibleDoublescats=[]
possibleDoublesMirrorcats=[]

for i in range(len(allcats)):
    for j in range(len(allcats[i])):
        for k in range(j+1,len(allcats[i])):
            if(os.path.getsize(allcats[i][j])== os.path.getsize(allcats[i][k])):
                possibleDoublescats.append(allcats[i][j])
                possibleDoublesMirrorcats.append(allcats[i][k])
        for u in range(len(cats_withoutrace)):
            if(os.path.getsize(allcats[i][j])== os.path.getsize(cats_withoutrace[u])):
                possibleDoublescats.append(allcats[i][j])
                possibleDoublesMirrorcats.append(cats_withoutrace[u])
    print(i)


# In[60]:


len(possibleDoubles),len(possibleDoubles)


# In[59]:


len(possibleDoublescats), len(possibleDoublesMirrorcats)


# In[89]:


os.path.getsize(possibleDoubles[1]),os.path.getsize(possibleDoublesMirror[1])


# In[90]:


im = matplotlib.pyplot.imread(possibleDoubles[2])
im2= matplotlib.pyplot.imread(possibleDoublesMirror[2])


# In[96]:


similar=[]
similar2=[]
for i in range(len(possibleDoubles)):
    im = matplotlib.pyplot.imread(possibleDoubles[i])
    im2= matplotlib.pyplot.imread(possibleDoublesMirror[i])
    if(im.shape==im2.shape):
        similar.append(possibleDoubles[i])
        similar2.append(possibleDoublesMirror[i])


# In[98]:


similarcats=[]
similar2cats=[]
for i in range(len(possibleDoublescats)):
    im = matplotlib.pyplot.imread(possibleDoublescats[i])
    im2= matplotlib.pyplot.imread(possibleDoublesMirrorcats[i])
    if(im.shape==im2.shape):
        similarcats.append(possibleDoublescats[i])
        similar2cats.append(possibleDoublesMirrorcats[i])


# In[158]:


similarfinal=[]
similarfinal2=[]
for i in range(len(similar)):
    im = matplotlib.pyplot.imread(similar[i])
    im2= matplotlib.pyplot.imread(similar2[i])
    dublura=True
    check=im==im2
    for i2 in range(check.shape[0]):
        for j in range(check.shape[1]):
            for k in range(check.shape[2]):
                if check[i2][j][k]==False:
                    dublura=False
                    break
            if dublura==False:
                break
        if dublura==False:
            break
                
    if dublura==True:
        similarfinal.append(similar[i])
        similarfinal2.append(similar2[i])
    print(i)


# In[165]:


similarfinalcats=[]
similarfinal2cats=[]
for i in range(len(similarcats)):
    im = matplotlib.pyplot.imread(similarcats[i])
    im2= matplotlib.pyplot.imread(similar2cats[i])
    dublura=True
    check=im==im2
    for i2 in range(check.shape[0]):
        for j in range(check.shape[1]):
            for k in range(check.shape[2]):
                if check[i2][j][k]==False:
                    dublura=False
                    break
            if dublura==False:
                break
        if dublura==False:
            break
                
    if dublura==True:
        similarfinalcats.append(similarcats[i])
        similarfinal2cats.append(similar2cats[i])


# In[166]:





# In[171]:


len(similarfinal),len(similarfinal2)


# In[172]:


len(similarfinalcats),len(similarfinal2cats)


# Found 62 doubles with dogs and 21 with cats and removed them
# 

# In[192]:


for each in similarfinal2:
    if os.path.exists(each):
          os.remove(each)
    else:
          print(each)

