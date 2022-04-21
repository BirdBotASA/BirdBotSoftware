# BirdBot Software
This is the repository for the BirdBotASA machine learning software. BirdBot software is now launching a soft alpha which will allow people to run the software using any computer and camera hardware. Users can connect their Algorand Wallet as well as name their camera to be awarded BIRDS tokens daily.


## **1. DOWNLOAD MINICONDA AND BIRDBOT FOLDER TO DESKTOP**

- DOWNLOAD AND INSTALL MINICONDA 3.8: https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Windows-x86_64.exe|
  - DOWNLOAD AND INSTALL MINICONDA 3.8 (MacOS Version): https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-MacOSX-x86_64.pkg
  - DOWNLOAD AND INSTALL MINICONDA 3.8 (Linux Version): https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh

- DOWNLOAD BIRDBOT AND PLACE ON DESKTOP: https://github.com/BirdBotASA/BirdBotWindows/archive/refs/heads/main.zip

## **2. OPEN MINICONDA THEN FOLLOW INSTRUCTIONS BELOW**

```
1.) Create a new miniconda enviornment called BirdBot with python 3.8 installed.

➡ conda create -n BirdBot python=3.8

2.) Activate the miniconda enviornment that you just created.

➡ conda activate BirdBot

3.) Install CUDA for GPU accesses, increases speed of machine learning (Optional)

➡ conda install cudatoolkit=10.1

4.) Install package requirements for BirdBotML.py and TensorFlow 2.3.1

➡ pip install -r requirements.txt

AFTER SUCCESSFUL INSTALL

5.) Run BirdBot Camera Software and attach webcam or DSLR camera.

➡ python "BirdBotML.py"

6.) Press the "Real-Time Mode" button to start the camera software. Press Q on screen to stop camera.
```

![image](https://user-images.githubusercontent.com/98153765/164117037-ed1b0ed4-93b2-4ae9-9a9c-9dc5feec9570.png)
