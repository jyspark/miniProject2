#Copyright JuliePark 2018

import os
import labeling as train_test_Data
import modeling as modeling
import numpy as np

def main(user_data):
	if user_data == '1':

		path = '.'
		path_Data = path + '/User_Data'

		user_folder = str(input("Is there 'User_Data' Folder? (1 for Yes, 0 for No). "))
		if user_folder == '0':
			#If folder does not exist, make folders
			if os.path.exists(path_Data) == False:
				os.mkdir(path_Data)
				os.mkdir(path_Data + '/User_Train')
				os.mkdir(path_Data + '/User_Test')
				print("Store your data in the folder 'User_Data'.")
				print("The images should be in the format of 'ImageName_1.jpg'")
				print("Restart the Program.")
				os._exit(0)

			elif os.path.exists(path_Data) == True:
				#Exit the program if the folder is existed
				print("Folder already existed. Run with the existed data ")
				pass

		elif user_folder == '1':
			user_data_name = str(input("Are the images in the format of ImageName_1.jpg? (1 for Yes, 0 for No). "))

			if user_data_name == '1':
				#run the program with the user data
				print("You are about to run the program with your own data.\n")
				print("WARNING: Make sure file_name for labeling.py and pred_label from modeling.py have been changed")

				user_train_dir = path_Data +'/User_Train'
				user_test_dir = path_Data + '/User_Test'

				#Get label for train and test data
				training_Data = train_test_Data.label_and_train(user_train_dir)
				testing_Data = train_test_Data.label_and_test(user_test_dir)

				tr_img = np.array([i[0] for i in training_Data]).reshape(-1, 128,128,1)
				tr_lbl = np.array([i[1] for i in training_Data])

				tt_img = np.array([i[0] for i in testing_Data]).reshape(-1, 128,128,1)
				tt_lbl = np.array([i[1] for i in testing_Data])

				#modeling and plotting the data
				data_modeling = modeling.modeling_plotting(tr_img,tr_lbl,testing_Data)
			
			elif user_data_name == '0':
				#change the data names manually using the code
				print("Use 'change_Names.py' to change the names of images and re-run the program")
				os._exit(0)
			else: 
				print("Wrong Input.")
				print("Going back to main. \n\n")
				os.system('python main.py')
				

	elif user_data == '0':
		#Run the program with my data
		print("\nThe program is running with the prepared data.\n")

		train_dir = '/media/sf_Assginment1/EC601/miniProject2/data/train4/'
		test_dir = '/media/sf_Assginment1/EC601/miniProject2/data/test4/'
		
		training_Data = train_test_Data.label_and_train(train_dir)
		testing_Data = train_test_Data.label_and_test(test_dir)

		tr_img = np.array([i[0] for i in training_Data]).reshape(-1, 128,128,1)
		tr_lbl = np.array([i[1] for i in training_Data])

		tt_img = np.array([i[0] for i in testing_Data]).reshape(-1, 128,128,1)
		tt_lbl = np.array([i[1] for i in testing_Data])

		data_modeling = modeling.modeling_plotting(tr_img,tr_lbl,testing_Data)

	else:
		print("Wrong Input. ")
		print("Going back to main.\n\n")
		os.system('python main.py')

	
if __name__ == '__main__':
	user_data = str(input("Do you want to run with your own data set? (1 for Yes, 0 for No). "))
	main(user_data)


