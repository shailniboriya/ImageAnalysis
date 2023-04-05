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

print("[INFO] Loading images ...")
class_name_folder_list = listdir(directory_root)
   
for class_name_folder in class_name_folder_list:
    print(f"[INFO] Processing {class_name_folder} ...")
    class_name_image_list = listdir(f"{directory_root}/{class_name_folder}/")     
    
    for imge in class_name_image_list[:200]:
        file = f"{directory_root}/{class_name_folder}/{imge}"
        if file.endswith(".jpg") == True or file.endswith(".JPG") == True or file.endswith(".jpeg") == True or file.endswith(".JPEG") == True or file.endswith(".png") == True or file.endswith(".PNG") == True:
            image_list.append(convert_image_to_array(file))
            label_list.append(class_name_folder)
print("[INFO] Image loading completed")

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
