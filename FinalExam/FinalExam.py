# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13LR3QgIcwTPnDOy-ul266Bcx12DEvRUF
"""


import streamlit as st
import tensorflow as tf
import keras
import numpy


@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('final_model1.h5')
  return model
model = load_model()
st.write("""
# Outfit Detection"""
)
file=st.file_uploader("Choose a Fashion Outfit from computer",type=["jpg","png"])

from PIL import Image,ImageOps
import numpy as np
import PIL
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model

def load_image(image):
	# convert to array
	img = img_to_array(image)
	# reshape into a single sample with 3 channels
	img = img.reshape(3, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image)
    img = img_to_array(image)
    # reshape into a single sample with 3 channels
    img = img.reshape(3, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    prediction = numpy.argmax(model.predict(img), axis=1)
    class_names=['T-shirt', 'Trouser', 'Pullover', 'Dress','Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Shoe']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
