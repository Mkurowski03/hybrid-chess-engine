@echo off
cd /d "E:\pycharm projects\chessbot"
call venv\Scripts\activate
python src\uci.py --sims 2200 --cpuct 1.25 --material 0.15 --discount 0.90
