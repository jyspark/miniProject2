import cv2
import os
import numpy as np
from random import shuffle
from tqdm import tqdm
import tensorflow as tf

#Labels the first set as 1,0 and the second set as 0,1
def labeling(img):
	
	file_name = img.split('_')[0]
	if file_name == 'FL':					#First Object name
		label = np.array([1,0])
	elif file_name == 'cocoa':				#Second Object name
		label = np.array([0,1])
	return label 

#read the images in grayscale & label and train the data set
def label_and_train(train_dir):
	
	train_set =[]	
	for i in tqdm(os.listdir(train_dir)):
		path = os.path.join(train_dir,i)
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (128,128))
		train_set.append([np.array(img), labeling(i)])
	shuffle(train_set)
	return train_set

#read the images in grayscale & label and test the data set
def label_and_test(test_dir):
	
	test_set =[]
	for i in tqdm(os.listdir(test_dir)):
		path = os.path.join(test_dir,i)
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (128,128))
		test_set.append([np.array(img), labeling(i)])
	shuffle(test_set)
	#print("test images:" ,test_images)
	return test_set