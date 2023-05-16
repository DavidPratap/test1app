import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
st.title("Dog and Cat Classifier Using Tensorflow and Keras")
model="dogs_cats_small.h5"
model=load_model(model)
uploaded_file=st.file_uploader("Upload a Cat or dog phote in jpeg")
for file in uploaded_file:
    file_path=uploaded_file.name
my_image=image.load_img(file_path, target_size=(150, 150))
my_image_array=image.img_to_array(my_image)
if st.checkbox("Display Image", False):
    image=Image.open(file_path)
    st.image(image)
my_image_array=np.expand_dims(my_image_array, axis=0)
prediction =int(model.predict(my_image_array)[0][0])
if st.button("Predict"):
    if prediction==0:
        st.subheader("Its a 🐱")
    if prediction==1:
        st.subheader("Its a 🐕")

