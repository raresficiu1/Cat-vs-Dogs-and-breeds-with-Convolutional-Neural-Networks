
# coding: utf-8

# In[1]:


import os
import glob
from sklearn.utils import shuffle
from collections import Counter
from shutil import copyfile,move
from sklearn.model_selection import train_test_split


# In[2]:


dog_dataset = glob.glob('/home/ficiu/Desktop/Train_CainePisica/Dog/*')
cat_dataset=glob.glob('/home/ficiu/Desktop/Train_CainePisica/Cat/*')
# cat_pictures = glob.glob('/home/ficiu/Desktop/Train_CainePisica/Cat/*')
dogs_without_race=glob.glob('/home/ficiu/Desktop/Train_data/Caini/Fara rasa/*')
cats_without_race=glob.glob('/home/ficiu/Desktop/Train_data/Pisici/Fara rasa/*')


# In[4]:


def appendingFunction(X_dataset,y_dataset,thelist):
    for each in thelist:
        for i in glob.glob(each+'*'):
            X_dataset.append(i)
            y_races.append(each.split('/')[-1])
    return X_dataset,y_dataset


# In[ ]:


X_cats_dogs=[]
y_cats_dogs=[]

X_races=[]
y_races=[]
z_races=[]

X_races,y_races=appendingFunction(X_races,y_races,dogs_races)
X_races,y_races=appendingFunction(X_races,y_races,cats_races)


# In[6]:


print(len(cat_dataset),len(dog_dataset))


# In[404]:


# X_races=X_races[:5]
# y_races=y_races[:5]


# In[ ]:


count_dogs=0
count_cats=0
X_cats=[]
X_dogs=[]
for each in races:
    for i in glob.glob(each+'/*'):
        X_dogs.append(i)
        y_cats_dogs.append('Dog')
        count_dogs+=1
        
for each in cats_races:
    for i in glob.glob(each+'/*'):
        X_cats.append(i)
        y_cats_dogs.append('Cat')
        count_cats+=1
        
for each in dogs_without_race:
    X_dogs.append(each)
    y_cats_dogs.append('Dog')
    count_dogs+=1

for each in cats_without_race:
    X_cats.append(each)
    y_cats_dogs.append('Cat')
    count_cats+=1
    
print(count_dogs,count_cats)
X_cats_dogs, y_cats_dogs = shuffle(X_cats_dogs, y_cats_dogs)
X_cats_dogs=X_cats_dogs
y_cats_dogs=y_cats_dogs


# In[406]:


count_dogs


# In[407]:


y_cats_dogs


# In[408]:


def echilibrate_dataset(X_batch1 ,y_batch1):
    X_batch=X_batch1.copy()
    y_batch=y_batch1.copy()
    X_batch.append('/home/ficiu/Desktop/Train_data/Caini/Cu rasa/asdasdadasdasdassdasd/8.jpg')
    y_batch.append('asdasdadasdasdassdasd')
    appearences=Counter(y_batch)
    howMany=0
    for each in appearences:
        if(appearences[each]>howMany):
            howMany=appearences[each]
    count=1
    locationPrev=0
    current=X_batch[0]
    for i in X_batch[1:]:
        current1=i
        prev=current.split('/')[-4]
        now=current1.split('/')[-4]
        print(now,prev)
        if(prev==now) and (current1!=X_batch[-1]):
            count+=1
        else :
            j=locationPrev
            while(count<howMany):
                X_batch.append(X_batch[j])
                y_batch.append(y_batch[j])
                j+=1
                count+=1
                if(j>=X_batch.index(current1)):
                    j=locationPrev
            
            current=current1
            locationPrev=X_batch.index(current1)
            count=1
    
    X_batch.remove('/home/ficiu/Desktop/Train_data/Caini/Cu rasa/asdasdadasdasdassdasd/8.jpg')
    y_batch.remove('asdasdadasdasdassdasd')
    return X_batch,y_batch


# In[409]:


# X_cats_dogs,y_cats_dogs=echilibrate_dataset(X_cats_dogs,y_cats_dogs)
# print(len(X_cats_dogs),len(y_cats_dogs))


# In[93]:


len(y_cats_dogs),len(X_cats_dogs)


# In[116]:


len(X_dogs)-len(X_cats)


# In[173]:


for each in cat_dataset[:16445]:
    copyfile(each, '/home/ficiu/Desktop/Train_cat_dog/Cat/1'+each.split('/')[-2]+each.split('/')[-1])


# In[97]:


X_cats[1]


# In[101]:


X_cats[1].split('/')[-2]


# In[167]:


dog_dataset = glob.glob('/home/ficiu/Desktop/Train_cat_dog/Dog/*')
cat_dataset=glob.glob('/home/ficiu/Desktop/Train_cat_dog/Cat/*')


# In[168]:


len(cat_dataset)


# In[160]:


for photo in dog_dataset[9686:14528]:
    move(photo,"/home/ficiu/Desktop/Test_Cat_Dog/Dog/"+photo.split('/')[-1])


# In[152]:


for photo in cat_dataset[4986:7479]:
    move(photo, "/home/ficiu/Desktop/Test_Cat_Dog/Cat/"+photo.split('/')[-1])


# In[172]:


dog_dataset=shuffle(dog_dataset)
cat_dataset=shuffle(cat_dataset)


# In[135]:


len(X_dogs)


# In[136]:


len(X_cats)


# In[170]:


len(dog_dataset)


# In[171]:


len(cat_dataset)


# In[169]:


len(dog_dataset)-len(cat_dataset)


# In[174]:


dog_dataset1 = glob.glob('/home/ficiu/Desktop/Validation_Cat_Dog/Dog/*')


# In[155]:


len(dog_dataset1)


# In[156]:


for photo in dog_dataset1:
    move(photo,"/home/ficiu/Desktop/Train_CainePisica/Dog/"+photo.split('/')[-1])


# In[159]:


9686+4842


# In[56]:


dogs_races = glob.glob('/home/ficiu/Desktop/Train_data/Caini/Cu rasa/*')
cats_races = glob.glob('/home/ficiu/Desktop/Train_data/Pisici/Cu rasa/*')
print(len(dogs_races),len(cats_races))


# In[57]:


def appendingFunction(X_dataset,y_dataset,thelist):
    for each in thelist:
        for i in glob.glob(each+'*'):
            X_dataset.append(i)
            y_races.append(each.split('/')[-1])
    return X_dataset,y_dataset


# In[58]:


X_races=[]
y_races=[]
z_races=[]

X_races,y_races=appendingFunction(X_races,y_races,dogs_races)
#X_races,y_races=appendingFunction(X_races,y_races,cats_races)


# In[59]:


print(len(X_races),len(y_races))


# In[60]:


y_races1=y_races[0]
for each in y_races[1:]:
    y_races1=y_races1+' '+each
print(y_races1)


# In[61]:


X_final=[]
y_final=[]
for each in X_races:
    list_pictures=glob.glob(each+'/*')
    for each1 in list_pictures:
        X_final.append(each1)
        y_final.append(each1.split('/')[-2])
Counter(y_final)


# In[62]:


X_final


# In[63]:


len(y_final)


# In[64]:


len(X_final)


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=(len(X_final)*20//100),stratify=None)


# In[66]:


len(X_train)


# In[67]:


X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=(len(X_final)*10//100),stratify=None)


# In[68]:


len(X_train),len(X_test),len(X_validation)


# In[69]:


len(y_train),len(y_test),len(y_validation)


# In[70]:


X_train[1].split('/')[-1]


# In[71]:


Counter(y_train)


# In[72]:


def echilibrate_dataset(X_batch1 ,y_batch1):
    X_batch=X_batch1.copy()
    y_batch=y_batch1.copy()
    X_batch.append('/home/ficiu/Desktop/Train_data/Caini/Cu rasa/asdasdadasdasdassdasd/8.jpg')
    y_batch.append('asdasdadasdasdassdasd')
    appearences=Counter(y_batch)
    howMany=0
    for each in appearences:
        if(appearences[each]>howMany):
            howMany=appearences[each]
    count=1
    locationPrev=0
    current=X_batch[0]
    print(howMany)
    for i in X_batch[1:]:
        current1=i
        prev=current.split('/')[-2]
        now=current1.split('/')[-2]
        if(prev==now) and (current1!=X_batch[-1]):
            count+=1
        else :
            j=locationPrev
            while(count<howMany):
                X_batch.append(X_batch[j])
                y_batch.append(y_batch[j])
                j+=1
                count+=1
                if(j>=X_batch.index(current1)):
                    j=locationPrev
            
            current=current1
            locationPrev=X_batch.index(current1)
            count=1
    X_batch.remove('/home/ficiu/Desktop/Train_data/Caini/Cu rasa/asdasdadasdasdassdasd/8.jpg')
    y_batch.remove('asdasdadasdasdassdasd')
    return X_batch,y_batch


# In[73]:


X_train


# In[74]:


Counter(y_validation)


# In[ ]:


os.mkdir('home/ficiu/Desktop/Validation_Dogs/')


# In[75]:


fisierTrain='/home/ficiu/Desktop/Train_Dogs/'
fisierTrain1='/home/ficiu/Desktop/Train_Dogs1/'
fisierTest='/home/ficiu/Desktop/Test_Dogs/'
fisierValidation='home/ficiu/Desktop/Validation_Dogs/'


# In[76]:


for pict in X_train:
    if os.path.exists(fisierTrain+pict.split('/')[-2]):
        copyfile(pict,fisierTrain+pict.split('/')[-2]+'/'+pict.split('/')[-1])
    else:   
        os.mkdir(fisierTrain+pict.split('/')[-2])
        copyfile(pict,fisierTrain+pict.split('/')[-2]+'/'+pict.split('/')[-1])

print("Now Test")
for pict in X_test:
    if os.path.exists(fisierTest+pict.split('/')[-2]):
        copyfile(pict,fisierTest+pict.split('/')[-2]+'/'+pict.split('/')[-1])
    else:   
        os.mkdir(fisierTest+pict.split('/')[-2])
        copyfile(pict,fisierTest+pict.split('/')[-2]+'/'+pict.split('/')[-1])


# In[79]:


print("Now Test")
for pict in X_validation:
    if os.path.exists(fisierValidation + pict.split('/')[-2]):
        copyfile(pict,fisierValidation+pict.split('/')[-2]+'/'+pict.split('/')[-1])
    else:
        os.makedirs(fisierValidation + pict.split('/')[-2])
        copyfile(pict,fisierValidation+pict.split('/')[-2]+'/'+pict.split('/')[-1])


# In[80]:


dogs_races = glob.glob(fisierTrain+'/*')


# In[81]:


dogs_races


# In[82]:


X_final=[]
y_final=[]
for each in dogs_races:
    list_pictures=glob.glob(each+'/*')
    for each1 in list_pictures:
        X_final.append(each1)
        y_final.append(each1.split('/')[-2])


# In[83]:


len(X_final),len(y_final)


# In[84]:


Counter(y_final)


# In[85]:


X_train,y_train=echilibrate_dataset(X_final,y_final)
print(len(X_train),len(y_train))


# In[86]:


Counter(y_train)


# In[87]:


a=['','0','00','000','0000','00000','00000','000000','0000000','00000000','000000000','0000000000']


# In[88]:


for pict in X_train:
    if os.path.exists(fisierTrain1+pict.split('/')[-2]):
        i=0
        b=fisierTrain1+pict.split('/')[-2]+'/'
        while os.path.isfile(b+a[i]+pict.split('/')[-1]):
            i+=1
        print(i,b+a[i]+pict.split('/')[-1])
        copyfile(pict,b+a[i]+pict.split('/')[-1])
    else:   
        os.mkdir(fisierTrain1+pict.split('/')[-2])
        copyfile(pict,fisierTrain1+pict.split('/')[-2]+'/'+pict.split('/')[-1])

