# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13LR3QgIcwTPnDOy-ul266Bcx12DEvRUF
"""
pip install tensorflow
import streamlit as st
import keras

@st.cache(allow_output_mutation=True)
def load_model():
  model=('final_model1.h5')
  return model
model = load_model()
st.write("""
# Outfit Detection"""
)
file=st.file_uploader("Choose a Fashion Outfit from computer",type=["jpg","png"])

from PIL import Image,ImageOps
import numpy as np
import PIL

def import_and_predict(image_data,model):
    size=(64,64)
    image= ImageOps.fit(image_data,size, PIL.Image.Resampling.LANCZOS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image)
    size=(64,64)
    image = ImageOps.fit(image,size, PIL.Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    class_names=['T-shirt', 'Trouser', 'Pullover', 'Dress','Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Shoe']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
