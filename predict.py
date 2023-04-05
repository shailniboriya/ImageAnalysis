from data_prep import *
from hyperparams import *
from model import *

import pandas as pd
import glob

# In[ ] Load the model

print("[INFO] Loading model...")
model= tf.keras.models.load_model(f'{curr_dir}/output/models/model')

# In[ ] Predict class of test data

testdir=f"{curr_dir}/input/test"
img_file = []
img_class = []
for img in glob.glob(f"{testdir}/*.jpg") + glob.glob(f"{testdir}/*.JPG") + glob.glob(f"{testdir}/*.jpeg") + glob.glob(f"{testdir}/*.JPEG") + glob.glob(f"{testdir}/*.png") + glob.glob(f"{testdir}/*.PNG"):
    img_file.append(img)
    imgarray=convert_image_to_array(img)
    #predicting the class of test data
    imgarray = np.expand_dims(imgarray, axis=0)
    y_predict = model.predict(imgarray)
    y_pred = (y_predict > 0.5)
    rounded_predictions = np.argmax(y_pred, axis=1)
    names=class_names[rounded_predictions]
    img_class.append(names)
    
# In[] Results

finaldf= pd.DataFrame( {'img_file': img_file,'img_class': img_class})
print(finaldf)
finaldf.to_csv(f"{curr_dir}/output/results/output.csv", encoding='utf-8', index= False)
