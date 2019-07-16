import tensorflow.keras
import pandas as pd
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input as mobile
from keras.applications.inception_v3 import preprocess_input as v3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import glob
from keras.models import load_model
from skimage import io, filters

testFile=os.getcwd()+'/testFiles/*'
testFileDogs=os.getcwd()+'/Dogs/*'
testFileCats=os.getcwd()+'/Cats/*'
testFileAll=os.getcwd()+'/All/*'
models=os.getcwd()+'/Models/'





print("Loading Model InceptionV3 cat/dogs")
Cat_Dogv3 = load_model(models+'Cat-Dog1.00inception.hdf5')
# print("Loading Model MobileNet cat/dogs")
Cat_DogMobileNet = load_model(models+'CatDog99%.hdf5')


print("Loading Model InceptionV3 Cats")
catv3 = cat_onlyv3=load_model(models+'CatsOnly0.86inception.hdf5')
print("Loading Model MobileNet Cats")
catmobile=load_model(models+'CatsOnly86%.hdf5')


print("Loading Model InceptionV3 Dogs")
dogv3 = load_model(models+'DogsOnly-0.74inception.hdf5')
print("Loading Model MobileNet Dogs")
dogmobile = load_model(models+'DogsOnly0.68.hdf5')


print("Loading Model InceptionV3 All")
allv3 = load_model(models+'AllRaces0.70inception.hdf5')
print("Loading Model MobileNet All")
allmobile = load_model(models+'AllRaces71%Augmented.hdf5')












classesCatDog=['Cat','Dog']
classesCats=['abyssinian','bengal','birman','bombay','britishshorthair','egyptianmau','mainecoon','persian','ragdoll','russianblue','siamese','sphynx']
classesAll=['abyssinian',
 'affenpinscher',
 'afghanhound',
 'africanhuntingdog',
 'airedale',
 'americanbulldog',
 'americanpitbullterrier',
 'americanstaffordshireterrier',
 'appenzeller',
 'australianterrier',
 'basenji',
 'basset',
 'beagle',
 'bedlingtonterrier',
 'bengal',
 'bernesemountaindog',
 'birman',
 'black-and-tancoonhound',
 'blenheimspaniel',
 'bloodhound',
 'bluetick',
 'bombay',
 'bordercollie',
 'borderterrier',
 'borzoi',
 'bostonbull',
 'bouvierdesflandres',
 'boxer',
 'brabancongriffon',
 'briard',
 'britishshorthair',
 'brittanyspaniel',
 'bullmastiff',
 'cairn',
 'cardigan',
 'chesapeakebayretriever',
 'chihuahua',
 'chow',
 'clumber',
 'cockerspaniel',
 'collie',
 'curly-coatedretriever',
 'dandiedinmont',
 'dhole',
 'dingo',
 'doberman',
 'egyptianmau',
 'englishcockerspaniel',
 'englishfoxhound',
 'englishsetter',
 'englishspringer',
 'entlebucher',
 'eskimodog',
 'flat-coatedretriever',
 'frenchbulldog',
 'germanshepherd',
 'germanshorthaired',
 'giantschnauzer',
 'goldenretriever',
 'gordonsetter',
 'greatdane',
 'greaterswissmountaindog',
 'greatpyrenees',
 'groenendael',
 'havanese',
 'ibizanhound',
 'irishsetter',
 'irishterrier',
 'irishwaterspaniel',
 'irishwolfhound',
 'italiangreyhound',
 'japanesechin',
 'japanesespaniel',
 'keeshond',
 'kelpie',
 'kerryblueterrier',
 'komondor',
 'kuvasz',
 'labradorretriever',
 'lakelandterrier',
 'leonberg',
 'lhasa',
 'mainecoon',
 'malamute',
 'malinois',
 'maltesedog',
 'mexicanhairless',
 'miniaturepinscher',
 'miniaturepoodle',
 'miniatureschnauzer',
 'newfoundland',
 'norfolkterrier',
 'norwegianelkhound',
 'norwichterrier',
 'oldenglishsheepdog',
 'otterhound',
 'papillon',
 'pekinese',
 'pembroke',
 'persian',
 'pomeranian',
 'pug',
 'ragdoll',
 'redbone',
 'rhodesianridgeback',
 'rottweiler',
 'russianblue',
 'saintbernard',
 'saluki',
 'samoyed',
 'schipperke',
 'scotchterrier',
 'scottishdeerhound',
 'sealyhamterrier',
 'shetlandsheepdog',
 'shibainu',
 'shih-tzu',
 'siamese',
 'siberianhusky',
 'silkyterrier',
 'soft-coatedwheatenterrier',
 'sphynx',
 'staffordshirebullterrier',
 'standardpoodle',
 'standardschnauzer',
 'sussexspaniel',
 'tibetanmastiff',
 'tibetanterrier',
 'toypoodle',
 'toyterrier',
 'vizsla',
 'walkerhound',
 'weimaraner',
 'welshspringerspaniel',
 'westhighlandwhiteterrier',
 'wheatenterrier',
 'whippet',
 'wire-hairedfoxterrier',
 'yorkshireterrier']

classesDogs=['affenpinscher',
 'afghanhound',
 'africanhuntingdog',
 'airedale',
 'americanbulldog',
 'americanpitbullterrier',
 'americanstaffordshireterrier',
 'appenzeller',
 'australianterrier',
 'basenji',
 'basset',
 'beagle',
 'bedlingtonterrier',
 'bernesemountaindog',
 'black-and-tancoonhound',
 'blenheimspaniel',
 'bloodhound',
 'bluetick',
 'bordercollie',
 'borderterrier',
 'borzoi',
 'bostonbull',
 'bouvierdesflandres',
 'boxer',
 'brabancongriffon',
 'briard',
 'brittanyspaniel',
 'bullmastiff',
 'cairn',
 'cardigan',
 'chesapeakebayretriever',
 'chihuahua',
 'chow',
 'clumber',
 'cockerspaniel',
 'collie',
 'curly-coatedretriever',
 'dandiedinmont',
 'dhole',
 'dingo',
 'doberman',
 'englishcockerspaniel',
 'englishfoxhound',
 'englishsetter',
 'englishspringer',
 'entlebucher',
 'eskimodog',
 'flat-coatedretriever',
 'frenchbulldog',
 'germanshepherd',
 'germanshorthaired',
 'giantschnauzer',
 'goldenretriever',
 'gordonsetter',
 'greatdane',
 'greaterswissmountaindog',
 'greatpyrenees',
 'groenendael',
 'havanese',
 'ibizanhound',
 'irishsetter',
 'irishterrier',
 'irishwaterspaniel',
 'irishwolfhound',
 'italiangreyhound',
 'japanesechin',
 'japanesespaniel',
 'keeshond',
 'kelpie',
 'kerryblueterrier',
 'komondor',
 'kuvasz',
 'labradorretriever',
 'lakelandterrier',
 'leonberg',
 'lhasa',
 'malamute',
 'malinois',
 'maltesedog',
 'mexicanhairless',
 'miniaturepinscher',
 'miniaturepoodle',
 'miniatureschnauzer',
 'newfoundland',
 'norfolkterrier',
 'norwegianelkhound',
 'norwichterrier',
 'oldenglishsheepdog',
 'otterhound',
 'papillon',
 'pekinese',
 'pembroke',
 'pomeranian',
 'pug',
 'redbone',
 'rhodesianridgeback',
 'rottweiler',
 'saintbernard',
 'saluki',
 'samoyed',
 'schipperke',
 'scotchterrier',
 'scottishdeerhound',
 'sealyhamterrier',
 'shetlandsheepdog',
 'shibainu',
 'shih-tzu',
 'siberianhusky',
 'silkyterrier',
 'soft-coatedwheatenterrier',
 'staffordshirebullterrier',
 'standardpoodle',
 'standardschnauzer',
 'sussexspaniel',
 'tibetanmastiff',
 'tibetanterrier',
 'toypoodle',
 'toyterrier',
 'vizsla',
 'walkerhound',
 'weimaraner',
 'welshspringerspaniel',
 'westhighlandwhiteterrier',
 'wheatenterrier',
 'whippet',
 'wire-hairedfoxterrier',
 'yorkshireterrier']


def predictDogandCat():
	predictions=[]
	pictures=glob.glob(testFile)
	count=len(pictures)
	i=0
	for each in pictures:
		image=io.imread(each)
		image1=v3(image)
		image2=mobile(image)
		image1=np.expand_dims(image1, axis=0)
		image2=np.expand_dims(image2, axis=0)
		print("Predicting image "+str(i)+"/" + str(count))
		prediction1=np.argmax(Cat_Dogv3.predict(image1))
		prediction2=np.argmax(Cat_DogMobileNet.predict(image2))
		if(prediction1==prediction2):
			predictions.append([classesCatDog[prediction1],each.split('/')[-1]])
		else:
			predictions.append(['Unsure',each.split('/')[-1]])
		i+=1
	

	printall(predictions)


def predictCats():
	predictions=[]
	pictures=glob.glob(testFileCats)
	count=len(pictures)
	i=0
	for each in pictures:
		image=io.imread(each)
		image1=v3(image)
		image2=mobile(image)
		image1=np.expand_dims(image1, axis=0)
		image2=np.expand_dims(image2, axis=0)
		print("Predicting image "+str(i)+"/" + str(count))
		prediction1=np.argmax(catv3.predict(image1))
		prediction2=np.argmax(catmobile.predict(image2))
		if(prediction1==prediction2):
			predictions.append([classesCats[prediction1],each.split('/')[-1]])
		else:
			predictions.append([classesCats[prediction1]+' or '+ classesCats[prediction2],each.split('/')[-1]])

		i+=1

	printall(predictions)

def predictDogs():
	predictions=[]
	pictures=glob.glob(testFileDogs)
	count=len(pictures)
	i=0
	for each in pictures:
		image=io.imread(each)
		image1=v3(image)
		image2=mobile(image)
		image1=np.expand_dims(image1, axis=0)
		image2=np.expand_dims(image2, axis=0)
		print("Predicting image "+str(i)+"/" + str(count))
		prediction1=np.argmax(dogv3.predict(image1))
		prediction2=np.argmax(dogmobile.predict(image2))
		if(prediction1==prediction2):
			predictions.append([classesDogs[prediction1],each.split('/')[-1]])
		else:
			predictions.append([classesDogs[prediction1]+' or '+ classesDogs[prediction2],each.split('/')[-1]])

		i+=1

	printall(predictions)


def predictAll():
	predictions=[]
	pictures=glob.glob(testFileDogs)
	count=len(pictures)
	i=0
	for each in pictures:
		image=io.imread(each)
		image1=v3(image)
		image2=mobile(image)
		image1=np.expand_dims(image1, axis=0)
		image2=np.expand_dims(image2, axis=0)
		print("Predicting image "+str(i)+"/" + str(count))
		prediction1=np.argmax(allv3.predict(image1))
		prediction2=np.argmax(allmobile.predict(image2))
		if(prediction1==prediction2):
			predictions.append([classesAll[prediction1],each.split('/')[-1]])
		else:
			predictions.append([classesAll[prediction1]+' or '+ classesAll[prediction2],each.split('/')[-1]])

		i+=1

	printall(predictions)


def model1():
	predictions=[]
	pictures=glob.glob(testFile)
	count=len(pictures)
	i=0
	for each in pictures:
		image=io.imread(each)
		image1=v3(image)
		image2=mobile(image)
		image1=np.expand_dims(image1, axis=0)
		image2=np.expand_dims(image2, axis=0)
		print("Predicting image "+str(i)+"/" + str(count))
		prediction1=np.argmax(Cat_Dogv3.predict(image1))
		prediction2=np.argmax(Cat_DogMobileNet.predict(image2))
		if(prediction1==prediction2):
			if(prediction1==0):
				prediction3=np.argmax(catv3.predict(image1))
				prediction4=np.argmax(catmobile.predict(image2))
				if(prediction3==prediction4):
					race=classesCats[prediction3]
				else:
					race=[classesCats[prediction3],classesCats[prediction4]]
			else:
				prediction3=np.argmax(dogv3.predict(image1))
				prediction4=np.argmax(dogmobile.predict(image2))
				if(prediction3==prediction4):
					race=classesDogs[prediction3]
				else:
					race=[classesDogs[prediction3],classesDogs[prediction4]]
			predictions.append([classesCatDog[prediction1],race,each.split('/')[-1]])

		else:
			predictions.append(['Unsure',each.split('/')[-1]])
		i+=1

	printall(predictions)

def printall (list):
	if(len(list)>1):
		for each in list:
			print(each)
	else:
		print(list)
decision1=0
while(int(decision1)<6):
	print ('                     		                         ')
	print ("Select the model")
	print ("1-Dog/Cats InceptionV3 99.6% + MobileNet 99.2%")
	print ("2-Cats Only")
	print ("3-Dogs Only")
	print ("4-All races 71%")
	print ("5-Final Model")
	print ("6-Exit")
	decision1=input()

	if(decision1=='1'):
		predictDogandCat()
	elif (decision1=='2'):
		predictCats()
	elif (decision1=='3'):
		predictDogs()
	elif (decision1=='4'):
		predictAll()
	elif (decision1=='5'):
		model1()
	else:
		print('Exit')

