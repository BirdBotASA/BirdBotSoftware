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
from turtle import bgcolor
import cv2
import numpy as np
import tensorflow as tf
import subprocess
import yolov3.twitch_speaker as TCI
from tkinter import *
from tkinter import filedialog
import ttkbootstrap as ttk
#from ttkbootstrap.constants import *
from yolov3.utils import detect_image, detect_realtime, detect_video, detect_ip_camera, Load_Yolo_model, detect_video_realtime_mp, generate_ml_data
from yolov3.twitch_listener import Bot
from yolov3.configs import *
from yolov3.wallet import *

yolo = Load_Yolo_model()
#detect_image(yolo, image_path, "./IMAGES/plate_1_detect.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_video(yolo, video_path, 'detected.mp4', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))

#detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), realtime=False)

cwd = os.getcwd()

def processVideo():
    video_path = filedialog.askopenfilename(initialdir = "./YOLO_Videos/", title = "Select a Video File",filetypes = (("all video format","*.mp4*"),("all files","*.*")))
    detect_video(yolo, video_path, video_path[:-4] + '-ML.mp4', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), HatMode=HAT_MODE)
	
def RealTimeMode():
    Wallet_Input = ALGORAND_WALLET_LABEL.get("1.0", "end-1c")
    Camera_Input = BIRDBOT_CAMERA_LABEL.get("1.0", "end-1c")
    Camera_ID_Input = current_value.get()
    Wallet = open(cwd + "\yolov3\wallet.py", "w")
    Wallet.write("BIRDBOT_CAMERA_NAME         = " + "\'" + Camera_Input + "\'" + "\n")
    Wallet.write("ALGORAND_WALLET             = " + "\'" + Wallet_Input + "\'")
    Wallet.close()
    detect_realtime(yolo, './YOLO_Videos/', Camera_ID_Input, input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0), ALGORAND_WALLET=Wallet_Input)
	
def IPCameraMode():
    detect_ip_camera(yolo, './YOLO_Videos/', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))
    
def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = indata

def exit():
	exit()


# Create Root Window
window = ttk.Window()
window.title('BirdBot ML Explorer')
window_width = 375
window_height = 800
window.resizable(False, False)

# Set Position of Window to Center of Screen
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
center_x = int(screen_width/2 - window_width / 2)
center_y = int(screen_height/2 - window_height / 2)
window.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

# Set Windows Icons
window.iconbitmap('logo.ico')

# BirdBot Logo
img = PhotoImage(file='logo_dash.png')
Label(window, image=img).pack(side=TOP)

#GUI Banner
header = Label(window, text = "BirdBot Alpha Software", font=("Roboto", 14), width = 100, height = 2)
header.config(bg='#212121', fg='#ffffff')
header.pack(pady=(0,10))

# Camera Name
camera_label = LabelFrame(window, text = "Camera Name (Optional)")
camera_label.config(font =("Roboto", 10))
camera_label.pack(side=TOP, pady=10, padx=10)

BIRDBOT_CAMERA_LABEL = Text(camera_label, height = 1, width = 60)
BIRDBOT_CAMERA_LABEL.insert(1.0, BIRDBOT_CAMERA_NAME)
BIRDBOT_CAMERA_LABEL.pack(side=TOP, pady=10, padx=10)

# Camera ID
current_value = ttk.StringVar(value=0)

# Algorand Wallet
algorand_label = LabelFrame(window, text = "Algorand Wallet (Optional)")
algorand_label.config(font =("Roboto", 10))
algorand_label.pack(side=TOP, pady=10, padx=10)

ALGORAND_WALLET_LABEL = Text(algorand_label, height = 2, width = 100)
ALGORAND_WALLET_LABEL.insert(1.0, ALGORAND_WALLET)
ALGORAND_WALLET_LABEL.pack(side=TOP, pady=(5,10), padx=10)

# Camera Mode Buttons

btn1 = Button(window, text="Process Video", font=("Roboto", 12), command = processVideo)
btn1.config(width = 100, height = 2, bg='#e5d04c', fg='#ffffff', activebackground="#eed195", activeforeground="#ffffff")
btn1.pack(side=TOP, pady=(10,0))

btn2 = Button(window, text="Real Time Mode", font=("Roboto", 12), command = RealTimeMode)
btn2.config(width = 100, height = 2, bg='#e5d04c', fg='#ffffff', activebackground="#eed195", activeforeground="#ffffff")
btn2.pack(side=TOP, pady=5)

btn3 = Button(window, text="IP Camera Mode", font=("Roboto", 12), command = IPCameraMode)
btn3.config(width = 100, height = 2, bg='#e5d04c', fg='#ffffff', activebackground="#eed195", activeforeground="#ffffff")
btn3.pack(side=TOP)

# Button Highlighting

def enterBtn1(e):
   btn1.config(background='#eed195', foreground= '#ffffff')
def leaveBtn1(e):
   btn1.config(background= '#e5d04c', foreground= '#ffffff')

def enterBtn2(e):
   btn2.config(background='#eed195', foreground= '#ffffff')
def leaveBtn2(e):
   btn2.config(background= '#e5d04c', foreground= '#ffffff')

def enterBtn3(e):
   btn3.config(background='#eed195', foreground= '#ffffff')
def leaveBtn3(e):
   btn3.config(background= '#e5d04c', foreground= '#ffffff')

btn1.bind('<Enter>', enterBtn1)
btn1.bind('<Leave>', leaveBtn1)
btn2.bind('<Enter>', enterBtn2)
btn2.bind('<Leave>', leaveBtn2)
btn3.bind('<Enter>', enterBtn3)
btn3.bind('<Leave>', leaveBtn3)

# GUI Footer
footer = Label(window, text = "BirdBot Technology LLC, Tacoma, WA", font=("Roboto", 8), width = 100, height = 2)
footer.config(bg='#212121', fg='#ffffff')
footer.pack(side=BOTTOM)

os.system('cls')

# subprocess.Popen("python yolov3/twitch_listener.py")

# Let the window wait for any events
window.mainloop()