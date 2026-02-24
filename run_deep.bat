@echo off
cd /d "E:\pycharm projects\chessbot"
call venv\Scripts\activate
python src\uci.py --sims 1200
