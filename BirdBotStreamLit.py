import streamlit as st
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import cv2
from yolov3.utils import *
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
DATA_URL = ('BirdBotStreamlitData.csv')
SPECIES_LIST_URL = ('BirdBotStreamlitSpecies.csv')
METRICS_URL = ('BirdBotStreamlitMetrics.csv')
SIGHTINGS_URL = ('BirdBotStreamlitSightings.csv')

CLASSES=TRAIN_CLASSES
input_size=YOLO_INPUT_SIZE
score_threshold=0.5
iou_threshold=0.3
rectangle_colors=''
image = ''
LABEL = ''

def start_realtime():
    yolo = Load_Yolo_model()
    Wallet = open("B:\BirdBot\BirdKeras\TensorFlow2\yolov3\wallet.py", "w")
    Wallet.write("BIRDBOT_CAMERA_NAME         = " + "\'" + Camera_Input + "\'" + "\n")
    Wallet.write("ALGORAND_WALLET             = " + "\'" + Wallet_Input + "\'"+ "\n")
    Wallet.write("IP_CAMERA_NAME             = " + "\'" + IP_Input + "\'")
    Wallet.close()
    detect_realtime(yolo, './YOLO_Videos/', camera_id=Camera_Number,  input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0), ALGORAND_WALLET=Wallet_Input)

def start_predict(predict):
    yolo = Load_Yolo_model()
    uploadColumn, predictColumn = st.columns(2)

    with uploadColumn:
        st.subheader('Uploaded Image')
        upload = Image.open(predict)
        with open(predict.name,"wb") as f:
            f.write(predict.getbuffer())
        st.image(upload)

    detect_image(yolo, predict.name, input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))

    with predictColumn:
        st.subheader('BirdBot Image')
        st.image("temp.jpg")
        time.sleep(1)
        os.remove(predict.name)

    with open('label.txt') as f:
        contents = f.read()
    st.subheader('BirdBot Prediction:')
    st.text(contents)
    st.markdown("""---""")

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
    
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
sightings_data = pd.read_csv(SIGHTINGS_URL)
metrics_data = pd.read_csv(METRICS_URL)

st.sidebar.title("BirdBot Settings Panel")
st.sidebar.header("Camera Settings")

with st.sidebar:
    Wallet_Input = st.text_input('Algorand Wallet', ALGORAND_WALLET)
    Camera_Input = st.text_input('Camera Name', BIRDBOT_CAMERA_NAME)
    IP_Input = st.text_input('IP Camera URL', IP_CAMERA_NAME)
    Camera_Number = st.number_input('Camera Number', step=1)
    container = st.container()
    st.markdown("""---""")
    st.write('Algorand Wallet:', Wallet_Input)
    st.write('Camera Name:', Camera_Input)
    st.write('IP URL:', IP_Input)
    st.write('Camera Number:', Camera_Number)

st.header("Detection Demos")

file = st.file_uploader('Upload An Image', type=['jpg', 'jpeg'])

if file:  # if user uploaded file    
        start_predict(file)

## REAL TIME MODE UNCOMMENT TO ACCESS ###

run = st.checkbox('Run webcam')
FRAME_WINDOW = st.image([])
vid = cv2.VideoCapture(Camera_Number)

if run:
    Yolo = Load_Yolo_model()

while run:
    _, frame = vid.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_data = image_preprocess(np.copy(frame), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    t1 = time.time()
	
    if YOLO_FRAMEWORK == "tf":
        pred_bbox = Yolo.predict(image_data)
	
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, frame, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')        
        
    frame = draw_bbox(frame, bboxes, CLASSES=CLASSES, draw_rect=True, rectangle_colors=rectangle_colors)
	
    FRAME_WINDOW.image(frame)

if container.button('Disconnect Camera'):
    cv2.destroyAllWindows()
    vid.release()
	
## REAL TIME MODE UNCOMMENT TO ACCESS ###
	
with st.expander("See supported objects"):
    species_data = pd.read_csv(SPECIES_LIST_URL)

    st.table(species_data)

st.markdown("""---""")

st.header('BirdBot Metrics')

st.subheader('Blockchain and Sightings Statistics')

TotalWallets, TotalTransactions, TotalBirdsRewarded = st.columns(3)

TotalWallets.metric("Wallets Connected", metrics_data['CountWallets'])
TotalTransactions.metric("Software Transactions Sent", metrics_data['CountTimesSentRewards'])
TotalBirdsRewarded.metric("Total BIRDS Awarded", metrics_data['SumTotalSentRewards'])

st.markdown("""---""")

TotalSightings, AverageSightings, MostSeenToday = st.columns(3)

MostSeenSpecies = str(metrics_data['MostSeenSpecies'])

TotalSightings.metric("Bird Sightings", metrics_data['CountTotalRows'])
AverageSightings.metric("Average Daily Bird Sightings", metrics_data['AverageSightings'])
MostSeenToday.metric("Today's Most Seen Species", MostSeenSpecies[3:])

st.markdown("""---""")

st.subheader('Wildlife Sightings by Hour - Global Data')

hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

# print(hist_values)

st.bar_chart(hist_values)

st.subheader('Wildlife Sightings by Species - Global Data')

sightings_species = sightings_data['BirdSpecies'].tolist()

# print(sightings_species)

sightings_count = sightings_data['Sightings'].tolist()

# print(sightings_count)

sightings_chart_data = pd.DataFrame(sightings_count, index=sightings_species)

print(sightings_chart_data)

st.bar_chart(sightings_chart_data)

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