@echo off
cd /d "E:\pycharm projects\chessbot"
call venv\Scripts\activate
python src\uci.py --model checkpoints\baseline\chessnet_epoch9.pt --book books\opening_book.bin --sims 40000 --cpuct 1.25 --material 0.15 --discount 0.90 --batch_size 256
