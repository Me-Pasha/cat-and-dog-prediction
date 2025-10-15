import streamlit as st
import joblib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

IMG_SIZE = (128, 128)
model = joblib.load("cnn_image_classification.pkl")
# Title of the app
st.title("Cat vs Dog Classifier 🐱🐶")


uploaded_file = st.file_uploader("Choose an image",  type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file)
    st.write("Image successfully uploaded ✅")
    img = load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 128, 128, 3)
    y_pred = model.predict(img_array)
    if y_pred>= 0.5:
        st.write("## 🐶 It's a **Dog**!")
    else:
        st.write("## 🐱 It's a **Cat**!")