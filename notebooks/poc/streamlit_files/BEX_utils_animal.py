import cv2
import pathlib 
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model

str_path = pathlib.Path(r'C:\Users\AndreasP\Desktop\work\BeyondExperience\streamlit_files\tiger.jpg')

def dummy_model(str_filename):
    comp_path = str_path 
    original_list = ["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]
    model = load_model('animal_model.h5')

    img = cv2.imread(str(str_filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = img.reshape(1, 224, 224, 3)
    pred = model.predict(img)
    species = np.argmax(pred)
    

    return original_list[species]

# z = dummy_model("tiger.jpg")
# print(z)