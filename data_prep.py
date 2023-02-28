from hyperparams import *

from sklearn.preprocessing import LabelBinarizer
from os import listdir
import cv2
import numpy as np
import tensorflow as tf

# load data

def convert_image_to_array(image_dir):
    try:
        imag = cv2.imread(image_dir)
        if imag is not None :
            imag = cv2.resize(imag, default_image_size)   
            return tf.keras.preprocessing.image.img_to_array(imag)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
        
# In[ ]: Fetch images from directory

image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for superclass_folder in root_dir :
        class_name_folder_list = listdir(f"{directory_root}/{superclass_folder}")
        
        for class_folder in class_name_folder_list :
            # remove .DS_Store from list
            if class_folder == ".DS_Store" :
                class_name_folder_list.remove(class_folder)

        for class_name_folder in class_name_folder_list:
            print(f"[INFO] Processing {class_name_folder} ...")
            class_name_image_list = listdir(f"{directory_root}/{superclass_folder}/{class_name_folder}/")
                
            for single_class_name_image in class_name_image_list :
                if single_class_name_image == ".DS_Store" :
                    class_name_image_list.remove(single_class_name_image)

            for imge in class_name_image_list[:200]:
                image_directory = f"{directory_root}/{superclass_folder}/{class_name_folder}/{imge}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    label_list.append(class_name_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")
    

# data preprocessing
# In[ ]:Get Size of Processed Image

image_size = len(image_list)
print("Image size is : ", image_size)

# In[ ]: Transform Image Labels uisng Scikit Learn's LabelBinarizer

label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
n_classes = len(label_binarizer.classes_)

# In[ ]:Print the classes

class_names=label_binarizer.classes_
print(class_names)
