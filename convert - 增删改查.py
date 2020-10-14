import pandas as pd
import shutil
import os

# this function is for read image,the input is directory name
def read_directory(directory_name,new_path):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(directory_name):
        if filename[9]>'4':
        #if filename1 == filename.split('.')[1]:
        #if filename1=='json':
            #os.remove(directory_name+filename)
            shutil.copy(os.path.join(directory_name, filename), os.path.join(new_path, filename))
            print(filename) #just for test
        #else:
        #    print("The file does not exist")
        #    print(filename)  # just for test
        #shutil.copy(os.path.join(directory_name, filename), os.path.join(new_path, filename))
        #print(filename)



if __name__ == '__main__':

    read_directory(r"F:\download\classification\weather_classification\haze\/","C:\\Users\\admin\\Desktop\\image_large\\train\\foggy\\")


