#================================================================
#
#   File name   : BirdBotML.py
#   Author      : Tyler Odenthal
#   Created date: 2020-08-18
#   
#   Website : https://www.bird.bot/
#   GitHub  : https://github.com/BirdBotASA/BirdBotWindows
#
#	Description : Basic TKinter GUI for running BirdBot Software
#
#================================================================

import os
import cv2
import numpy as np
import tensorflow as tf
import subprocess
import yolov3.twitch_speaker as TCI
from tkinter import *
from tkinter import filedialog
from yolov3.utils import detect_image, detect_realtime, detect_video, detect_ip_camera, Load_Yolo_model, detect_video_realtime_mp, generate_ml_data
from yolov3.twitch_listener import Bot
from yolov3.configs import *
from yolov3.wallet import *

yolo = Load_Yolo_model()
#detect_image(yolo, image_path, "./IMAGES/plate_1_detect.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_video(yolo, video_path, 'detected.mp4', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))

#detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), realtime=False)

def processVideo():
    video_path = filedialog.askopenfilename(initialdir = "./YOLO_Videos/", title = "Select a Video File",filetypes = (("all video format","*.mp4*"),("all files","*.*")))
    detect_video(yolo, video_path, video_path[:-4] + '-ML.mp4', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), HatMode=HAT_MODE)
	
def RealTimeMode():
    Wallet_Input = ALGORAND_WALLET_LABEL.get("1.0", "end-1c")
    Camera_Input = BIRDBOT_CAMERA_LABEL.get("1.0", "end-1c")
    Wallet = open("B:\BirdBot\BirdKeras\TensorFlow2\yolov3\wallet.py", "w")
    Wallet.write("BIRDBOT_CAMERA_NAME         = " + "\'" + Camera_Input + "\'" + "\n")
    Wallet.write("ALGORAND_WALLET             = " + "\'" + Wallet_Input + "\'")
    Wallet.close()
    detect_realtime(yolo, './YOLO_Videos/', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0), ALGORAND_WALLET=Wallet_Input)
	
def IPCameraMode():
    detect_ip_camera(yolo, './YOLO_Videos/', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))
    
def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = indata

def exit():
	exit()
    
# Create the root window
window = Tk()

# Set window title
window.title('BirdBot ML Software Explorer')

# Set window size
window.geometry("700x500")

#Set window background color
window.config(background = "white")

# Create a File Explorer label
label_file_explorer = Label(window,
                            text = "BirdBot Alpha ML Software - Please Use Buttons Below!",
                            width = 100, height = 4,
                            fg = "blue")     

ALGORAND_WALLET_LABEL = Text(window, height = 1, width = 60)
ALGORAND_WALLET_LABEL.insert(1.0, ALGORAND_WALLET)
 
# Create label
algorand_label = Label(window, text = "Algorand Wallet - Optional")
algorand_label.config(font =("Courier", 10))

BIRDBOT_CAMERA_LABEL = Text(window, height = 1, width = 60)
BIRDBOT_CAMERA_LABEL.insert(1.0, BIRDBOT_CAMERA_NAME)
 
# Create label
camera_label = Label(window, text = "Camera Name - Optional")
camera_label.config(font =("Courier", 10))

button_process_video = Button(window,
                        text = "Process Video",
                        command = processVideo)
                        
button_realtime_mode = Button(window,
                        text = "Real-Time Mode",
                        command = RealTimeMode)
						
button_ip_camera_mode = Button(window,
                        text = "IP Camera Mode",
                        command = IPCameraMode)

button_exit = Button(window,
                    text = "Exit",
                    command = exit)

# Grid method is chosen for placing
# the widgets at respective positions
# in a table like structure by
# specifying rows and columns
label_file_explorer.grid(row = 0, column = 0, pady = 10)

algorand_label.grid(row = 1, column = 0, columnspan = 2, pady = 5)

ALGORAND_WALLET_LABEL.grid(row = 2, column = 0, columnspan = 2, pady = 5)

camera_label.grid(row = 3, column = 0, columnspan = 2, pady = 5)

BIRDBOT_CAMERA_LABEL.grid(row = 4, column = 0, columnspan = 2, pady = 5)

button_process_video.grid(row = 5, column = 0, columnspan = 2, pady = 5)

button_realtime_mode.grid(row = 6, column = 0, columnspan = 2, pady = 5)

button_ip_camera_mode.grid(row = 7, column = 0, columnspan = 2, pady = 5)

button_exit.grid(row = 8, column = 0, columnspan = 2, pady = 5)

os.system('cls')

# subprocess.Popen("python yolov3/twitch_listener.py")

# Let the window wait for any events
window.mainloop()