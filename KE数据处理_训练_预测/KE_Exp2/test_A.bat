@echo off

REM ����Python��������·��
call D:\ProgramData\anaconda3\Scripts\activate.bat activate pytorch

REM ����ѵ���ű�����
set GROUP=GroupA
set DATA_X=Data/test_A_X.csv
set MODEL_PATH=Models/A_900.pt
set CUDA_ABLE=--cuda_able
set THRESHOLD=0.6


REM ִ��ѵ������
python test.py --Group %GROUP% ^
                  --data_X %DATA_X% ^
                  --model_path %MODEL_PATH% ^
                  --threshold %THRESHOLD% ^
                  %CUDA_ABLE%

pause
