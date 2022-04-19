# BirdBotWindows
This is the repository for the BirdBotASA machine learning software. A majority of this code is for machine learning set up, the rest is probably GUI files.



```
‣ install CUDA for GPU accesses, increases speed of machine learning (Optional)
conda create -n BirdBot python=3.8

‣ install CUDA for GPU accesses, increases speed of machine learning (Optional)
conda activate BirdBot

‣ install CUDA for GPU accesses, increases speed of machine learning (Optional)
conda install cudatoolkit=10.1

‣ install package requirements and TensorFlow 2.3.1
pip install -r ./requirements.txt

‣ yolov3
wget -P model_data https://pjreddie.com/media/files/yolov3.weights

‣ yolov3-tiny
wget -P model_data https://pjreddie.com/media/files/yolov3-tiny.weights

‣ yolov4
wget -P model_data https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

‣ yolov4-tiny
wget -P model_data https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

AFTER SUCCESSFUL INSTALL

➡ python "BirdBotML.py"
```
