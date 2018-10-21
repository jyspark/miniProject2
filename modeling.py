import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import matplotlib.pyplot as plt
from labeling import labeling

def modeling_plotting(tr_img,tr_lbl,testing_data):
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
	model.add(Dense(500,activation='relu'))				
	model.add(Dropout(rate=0.5))
	model.add(Dense(2,activation='softmax'))			


	model.compile(optimizer= Adam(lr=1e-3), loss='categorical_crossentropy',metrics=['accuracy'])
	fitting = model.fit(x=tr_img,y=tr_lbl,epochs=80,batch_size=100) #, validation_split=0.2)	 
	model.summary()

	'''plt.figure()
	plt.plot(fitting.history['acc'])
	plt.plot(fitting.history['val_acc'])
	plt.title('Model Accuracy')
	plt.ylabel('Acc')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig('Model Accuracy.png')

	plt.figure()
	plt.plot(fitting.history['loss'])
	plt.plot(fitting.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig('Model Loss.png')'''


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
			print (count+"th Result in Progress") 

		y = fig.add_subplot(6,5,c+1)

		img = data[0]
		data = img.reshape(1,128,128,1)
		predict_model = model.predict([data])

		if np.argmax(predict_model) == 1:
			pred_label = 'FL'				#Second Object name
		else:
			pred_label='cocoa'				#First Object name

		y.imshow(img, cmap='gray')

		plt.title(pred_label)
		y.axes.get_xaxis().set_visible(False)
		y.axes.get_yaxis().set_visible(False)

		plt.savefig('result.png')
	print("The result has been saved in the folder. ")

