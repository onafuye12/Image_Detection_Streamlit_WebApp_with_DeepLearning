
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from tensorflow.keras.utils import load_img, img_to_array
#from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model

model = load_model('best_model.h5')

classes = {0 : 'Bulldog', 1 : 'German Shepherd'}

img_file = st.file_uploader('select an image', type=['jpg','png','jpeg','gif','jfif','heic'])

if img_file is not None :
    img = Image.open(img_file)
    st.image(img,caption='Upload image succesfully')
    
    if st.button('predict'):
        img = img.resize((256,256))
        i = img_to_array(img)
        i = preprocess_input(i)
        input_arr = np.array([i])
        
        y_out = np.argmax(model.predict(input_arr))
        y_out1 = classes[y_out]
        
        #if y_out1 = 0:
        st.write(f'This image is a {y_out1}')
        #else:
            #st.write(f'This image is a {y_out1}')
