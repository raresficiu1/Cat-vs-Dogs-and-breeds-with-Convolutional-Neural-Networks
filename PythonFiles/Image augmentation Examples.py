
# coding: utf-8

# In[ ]:


from imgaug import augmenters as iaa
import numpy as np
from skimage import io


# In[86]:


augmenter=iaa.Sequential([iaa.Fliplr(1.0),iaa.Flipud(1),iaa.GaussianBlur(3.0),iaa.Affine(translate_px={"x": -150},mode='edge'),
                         iaa.AddToHueAndSaturation(value=45)

            ])


# In[156]:


clas='9'
picture=1
imageOriginal=io.imread('/home/ficiu/Desktop/85.jpg')


# In[157]:



io.imshow(imageOriginal)
skimage.io.imsave("/home/ficiu/Desktop/Raport/Exemple poze augmentate/"+clas+str(picture)+".jpg",imageOriginal)
picture+=1


# In[158]:


image=augmenter.augment_image(imageOriginal)
io.imshow(image)
skimage.io.imsave("/home/ficiu/Desktop/Raport/Exemple poze augmentate/"+clas+str(picture)+".jpg",image)
picture+=1


# In[159]:


flipper = iaa.Fliplr(1.0) # always horizontally flip each input image
image = flipper.augment_image(imageOriginal) # horizontally flip image 0
io.imshow(image)
skimage.io.imsave("/home/ficiu/Desktop/Raport/Exemple poze augmentate/"+clas+str(picture)+".jpg",image)
picture+=1


# In[168]:


vflipper = iaa.Rot90(3) # vertically flip each input image with 90% probability
image = vflipper.augment_image(imageOriginal) # probably vertically flip image 1
io.imshow(image)
skimage.io.imsave("/home/ficiu/Desktop/Raport/Exemple poze augmentate/"+clas+str(picture)+".jpg",image)
picture+=1


# In[161]:


blurer = iaa.GaussianBlur(3.0)
image = blurer.augment_image(imageOriginal)
io.imshow(image)
skimage.io.imsave("/home/ficiu/Desktop/Raport/Exemple poze augmentate/"+clas+str(picture)+".jpg",image)
picture+=1


# In[162]:


translater = iaa.Affine(translate_px={"x": -150},mode='edge') # move each input image by 16px to the left
image = translater.augment_image(imageOriginal) # move image 4 to the left
io.imshow(image)
skimage.io.imsave("/home/ficiu/Desktop/Raport/Exemple poze augmentate/"+clas+str(picture)+".jpg",image)
picture+=1


# In[163]:


scaler = iaa.AddToHueAndSaturation(value=45) # scale each input image to 80-120% on the y axis
image = scaler.augment_image(imageOriginal) # scale image
io.imshow(image)
skimage.io.imsave("/home/ficiu/Desktop/Raport/Exemple poze augmentate/"+clas+str(picture)+".jpg",image)
picture+=1


# In[164]:


scaler = iaa.AddToHueAndSaturation(value=-80) # scale each input image to 80-120% on the y axis
image = scaler.augment_image(imageOriginal) # scale image
io.imshow(image)
skimage.io.imsave("/home/ficiu/Desktop/Raport/Exemple poze augmentate/"+clas+str(picture)+".jpg",image)
picture+=1


# In[165]:


scaler = iaa.Affine(scale=(0.5, 0.7),mode='constant') # scale each input image to 80-120% on the y axis
image = scaler.augment_image(imageOriginal) # scale image
io.imshow(image)
skimage.io.imsave("/home/ficiu/Desktop/Raport/Exemple poze augmentate/"+clas+str(picture)+".jpg",image)
picture+=1


# In[166]:


scaler = iaa.Affine(scale=(0.5, 0.7),mode='edge') # scale each input image to 80-120% on the y axis
image = scaler.augment_image(imageOriginal) # scale image
io.imshow(image)
skimage.io.imsave("/home/ficiu/Desktop/Raport/Exemple poze augmentate/"+clas+str(picture)+".jpg",image)
picture+=1

