import streamlit as st
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, detect_ip_camera, Load_Yolo_model, detect_video_realtime_mp, generate_ml_data
from yolov3.configs import *
from yolov3.wallet import *
from PIL import Image
import os

st.set_page_config(
page_title="BirdBot - Machine Learning dApp",
page_icon="BirdBotIcon.png",
initial_sidebar_state="auto")

st.title("BirdBot - Machine Learning dApp")

DATE_COLUMN = 'date/time'
SPECIES_COLUMN = 'Bird_Species_Triggered'
DATA_URL = ('BirdBotStreamLitData.csv')
image = ''
LABEL = ''

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
    
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done!")
time.sleep(1)
data_load_state.text("")

file = st.file_uploader('Upload An Image', type=['jpg', 'jpeg'])

st.markdown("""---""")

if file:  # if user uploaded file

        col1, col2 = st.columns(2)

        yolo = Load_Yolo_model()

        with col1:
            st.subheader('Uploaded Image')
            upload = Image.open(file)
            with open(file.name,"wb") as f:
                f.write(file.getbuffer())
            st.image(upload)

        detect_image(yolo, file.name, input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))

        with col2:
            st.subheader('BirdBot Image')
            st.image("temp.jpg")
            time.sleep(1)
            os.remove(file.name)

        with open('label.txt') as f:
            contents = f.read()
        st.subheader('BirdBot Prediction:')
        st.text(contents)
        st.markdown("""---""")

st.subheader('Number of Wildlife Sightings by Hour - Global Data')

hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

st.bar_chart(hist_values)

st.markdown("""---""")

st.subheader('Use Slider to Adjust Time of Day')
hour_to_filter = st.slider('Hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all Wildlife Sightings at {hour_to_filter}:00')

st.map(filtered_data)

st.markdown("""---""")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)