import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
  model=('final_model1.hdf5')
  return model
model = load_model()
st.write("""
# Outfit Detection"""
)
file=st.file_uploader("Choose a Fashion Outfit from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(3, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

if file is None:
    st.text("Please upload an image file")
else:
    img = load_image(file)
    st.image(img,use_column_width=True)
    result = numpy.argmax(model.predict(img), axis=1)
    return result
    class_names=['T-shirt', 'Trouser', 'Pullover', 'Dress','Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Shoe']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
