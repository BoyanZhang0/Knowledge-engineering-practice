@echo off

REM ����Python��������·��
call D:\ProgramData\anaconda3\Scripts\activate.bat activate pytorch

REM ����ѵ���ű�����
set GROUP=GroupA
set DATA_X=Data\train_A_X.csv
set DATA_Y=Data\train_A_Y.csv
set EPOCHS=900
set LR=0.0001
set VALIDATE_EVERY=20
set REDIVIDE_EVERY=2
set CUDA_ABLE=--cuda_able
set THRESHOLD=0.6


REM ִ��ѵ������
python train.py --Group %GROUP% ^
                  --data_X %DATA_X% ^
                  --data_Y %DATA_Y% ^
                  --epochs %EPOCHS% ^
                  --lr %LR% ^
                  --validate_every %VALIDATE_EVERY% ^
                  --redivide_every %REDIVIDE_EVERY% ^
                  --threshold %THRESHOLD% ^
                  %CUDA_ABLE%

pause
