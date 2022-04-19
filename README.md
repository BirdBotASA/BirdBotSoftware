# BirdBotWindows
This is the repository for the BirdBotASA machine learning software. BirdBot software is now launching a soft alpha which will allow people to run the software using any computer and camera hardware. Users can connect their Algorand Wallet as well as name their camera to be awarded BIRDS tokens daily.

DOWNLOAD AND INSTALL MINICONDA 3.8: https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Windows-x86_64.exe

DOWNLOAD BIRDBOT AND PLACE ON DESKTOP: https://github.com/BirdBotASA/BirdBotWindows/archive/refs/heads/main.zip

```
OPEN MINICONDA THEN FOLLOW INSTRUCTIONS BELOW

1.) Create a new miniconda enviornment called BirdBot with python 3.8 installed.

➡ conda create -n BirdBot python=3.8

2.) Activate the miniconda enviornment that you just created.

➡ conda activate BirdBot

3.) Install CUDA for GPU accesses, increases speed of machine learning (Optional)

➡ conda install cudatoolkit=10.1

4.) Install package requirements for BirdBotML.py and TensorFlow 2.3.1

➡ pip install -r ./requirements.txt

AFTER SUCCESSFUL INSTALL

➡ python "BirdBotML.py"
```
