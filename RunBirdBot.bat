@echo off
call B:\miniconda3\Scripts\activate.bat
call cd "B:\BirdBot\BirdBotGitHub\BirdBotWindows"
call conda activate object2
call streamlit run BirdBotStreamLit.py
pause