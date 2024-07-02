@echo off

REM 设置Python环境变量路径
call D:\ProgramData\anaconda3\Scripts\activate.bat activate pytorch

REM 设置训练脚本参数
set GROUP=GroupA
set DATA_X=Data/test_A_X.csv
set MODEL_PATH=Models/A_900.pt
set CUDA_ABLE=--cuda_able
set THRESHOLD=0.6


REM 执行训练命令
python test.py --Group %GROUP% ^
                  --data_X %DATA_X% ^
                  --model_path %MODEL_PATH% ^
                  --threshold %THRESHOLD% ^
                  %CUDA_ABLE%

pause
