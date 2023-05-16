import streamlit as st
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Dog and Cat Classifier using TensorFlow and Keras")
model='dogs_and_cats_small.h5'
#image_path="C:/Users/David/Desktop/ML OPs/demo/dog-puppy-on-garden-royalty-free-image-1586966191.jpg"
#image_path=st.text_input("image path enter here", 'my_pet.jpg')
#image_path="C:/Users/David/Desktop/ML OPs/demo/catto.jpeg"
model=load_model(model)
from tensorflow.keras.preprocessing import image
uploaded_file=st.file_uploader("Upload a cat or dog photo")
for file in uploaded_file:
    file_path=uploaded_file.name
my_image=image.load_img(file_path, target_size=(150, 150))
my_img_arr=image.img_to_array(my_image)
if st.checkbox("Display Image", False):
    image=Image.open(file_path)
    st.image(image)
import numpy as np
my_img_arr=np.expand_dims(my_img_arr, axis=0)
prediction=int(model.predict(my_img_arr)[0][0])
if st.button("Predict"):
    if prediction==0:
        st.write("Its a Cat")
    if prediction==1:
        st.write('Its a dog')
