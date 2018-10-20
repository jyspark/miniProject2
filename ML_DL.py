#Copyright JuliePark 2018

import tensorflow as tf
import cv2
import os
import numpy as np
import keras
 
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt 

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *


#Path to data set for training and testing images
train_dir = '/media/sf_Assginment1/EC601/miniProject2/data/train4/'
test_dir = '/media/sf_Assginment1/EC601/miniProject2/data/test4/'

#Labels the first set as 1,0 and the second set as 0,1
def labeling(img):
	file_name = img.split('_')[0]
	if file_name == 'FL':					#car = sunkist
		label = np.array([1,0])
	elif file_name == 'cocoa':				#fruitella = truck
		label = np.array([0,1])
	return label 

#read the images in grayscale & label and train the data set
def label_and_train():
	train_set =[]
	for i in tqdm(os.listdir(train_dir)):
		path = os.path.join(train_dir,i)
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		#b,g,r = cv2.split(img)
		#img = cv2.merge([r,g,b])
		img = cv2.resize(img, (128,128))
		train_set.append([np.array(img), labeling(i)])
	shuffle(train_set)
	return train_set

#read the images in grayscale & label and test the data set
def label_and_test():
	test_set =[]
	for i in tqdm(os.listdir(test_dir)):
		path = os.path.join(test_dir,i)
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		#b,g,r = cv2.split(img)
		#img = cv2.merge([r,g,b])
		img = cv2.resize(img, (128,128))
		test_set.append([np.array(img), labeling(i)])
	shuffle(test_set)
	#print("test images:" ,test_images)
	return test_set


training_data = label_and_train()
testing_data = label_and_test()
#print("Testing_img: " , testing_img)
#print("Testing_img from 10 to 40?: ", testing_img[0:20])

#study the images
tr_img = np.array([i[0] for i in training_data]).reshape(-1, 128,128,1)
tr_lbl = np.array([i[1] for i in training_data])

tt_img = np.array([i[0] for i in testing_data]).reshape(-1, 128,128,1)
tt_lbl = np.array([i[1] for i in testing_data])

model = Sequential()

model.add(InputLayer(input_shape=[128,128,1]))
model.add(Conv2D(filters=32,kernel_size=5,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters=50,kernel_size=5,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters=80,kernel_size=5,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(500,activation='relu'))				#lower number = ,higher number
model.add(Dropout(rate=0.5))
model.add(Dense(2,activation='softmax'))			#2 data sets

model.compile(optimizer= Adam(lr=1e-3), loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=tr_img,y=tr_lbl,epochs=80,batch_size=100)		#increasing epochs=     , decreasing epochs= 
model.summary()

fig=plt.figure(figsize=(14,14))

#Display the result
for c, data in enumerate(testing_data[10:40]):
	
	if (c+1) == 1:
		print("1st Result in Progress ")
	elif (c+1) == 2:
		print("2nd Result in Progress ")
	elif (c+1) == 3:
		print("3rd Result in Progress ")
	else:
		count =str(c+1)
		print (count+"th Result in Progress") #c+1,"th Result in Progress ")

	y = fig.add_subplot(6,5,c+1)

	img = data[0]
	data = img.reshape(1,128,128,1)
	predict_model = model.predict([data])

	#print("model_out: " ,predict_model)
	
	if np.argmax(predict_model) == 1:
		pred_label = 'cocoa'				#truck = fruitella
	else:
		pred_label= 'FL'					# car == sunkist

	y.imshow(img, cmap='gray')

	plt.title(pred_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)

	plt.savefig('result3.png')
