import glob
import os

#Path to a folder containing images/files
path = '/media/sf_Assginment1/EC601/miniProject2/hh/'


#get file names which ends with .jpg and change to a new name
for i, filename in enumerate(glob.glob(path + '*.jpg')):
	os.rename(filename, os.path.join(path, 'cocoa_' + str(i+1) + '.jpg'))