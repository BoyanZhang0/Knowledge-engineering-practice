@echo off

REM 设置Python环境变量路径
set PYTHON=python

REM 设置训练脚本参数
set GROUP=GroupA
set DATA_X=Data\train_A_X.csv
set DATA_Y=Data\train_A_Y.csv
set EPOCHS=50000
set LR=0.000005
set VALIDATE_EVERY=20
set REDIVIDE_EVERY=2
set CUDA_ABLE=--cuda_able
set PRETRAIN=--pretrain
set PREMODEL=Models/GroupA_30000ep_2024-04-02_22-46-23.pt
set THRESHOLD=0.7

REM 执行训练命令
%PYTHON% train.py --Group %GROUP% ^
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
