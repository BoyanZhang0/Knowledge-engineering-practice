@echo off

REM 设置Python环境变量路径
call D:\ProgramData\anaconda3\Scripts\activate.bat activate pytorch

REM 设置训练脚本参数
set GROUP=GroupB
set DATA_X=Data\train_B_X.csv
set DATA_Y=Data\train_B_Y.csv
set EPOCHS=1000
set LR=0.0001
set VALIDATE_EVERY=20
set REDIVIDE_EVERY=2
set CUDA_ABLE=--cuda_able
set PRETRAIN=--pretrain
set PREMODEL=Models/B_9000.pt
set THRESHOLD=0.6


REM 执行训练命令
python train.py --Group %GROUP% ^
                  --data_X %DATA_X% ^
                  --data_Y %DATA_Y% ^
                  --epochs %EPOCHS% ^
                  --lr %LR% ^
                  --validate_every %VALIDATE_EVERY% ^
                  --redivide_every %REDIVIDE_EVERY% ^
                  %CUDA_ABLE% ^
                  %PRETRAIN% ^
                  --threshold %THRESHOLD% ^
                  --preModel %PREMODEL%

pause
