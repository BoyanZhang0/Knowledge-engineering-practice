import pandas as pd
import numpy as np
import re

# 根据测试数据的分组情况，将真实结果也分组便于之后评估


# 导入测试数据结果并处理
file_med = 'testdata/medicine.csv'
file_A = 'testdata/test_A.csv'
file_B = 'testdata/test_B.csv'
file_Y = 'testdata/real_Y.xlsx'

data_med = pd.read_csv(file_med, encoding='utf_8_sig')
data_A = pd.read_csv(file_A, encoding='utf_8_sig')
data_B = pd.read_csv(file_B, encoding='utf_8_sig')
data_Y = pd.read_excel(file_Y)

med = data_med.values[:, 1].tolist()
ID_A = data_A.values[:, 1]
ID_B = data_B.values[:, 1]
real_Y = data_Y.values[:, :]

real_A_Y = []
real_B_Y = []
real_A_N = []
real_B_N = []
N = []

for i in range(real_Y.shape[0]):
    id = real_Y[i][0]
    m = np.zeros(100)
    n = 0
    if id in ID_A:
        for j in range(1, real_Y.shape[1]):
            if real_Y[i][j] != real_Y[i][j]:
                break
            n += 1
            if real_Y[i][j] in med and m[med.index(real_Y[i][j])] == 0:
                m[med.index(real_Y[i][j])] = 1
        real_A_Y.append(m)
        real_A_N.append(n)
        N.append(n)
    elif id in ID_B:
        for j in range(1, real_Y.shape[1]):
            if real_Y[i][j] != real_Y[i][j]:
                break
            n += 1
            if real_Y[i][j] in med and m[med.index(real_Y[i][j])] == 0:
                m[med.index(real_Y[i][j])] = 1
        real_B_Y.append(m)
        real_B_N.append(n)
        N.append(n)
    else:
        print('error')


real_A_Y = pd.DataFrame(real_A_Y)
real_B_Y = pd.DataFrame(real_B_Y)
real_A_N = pd.DataFrame(real_A_N)
real_B_N = pd.DataFrame(real_B_N)

real_A_Y.to_csv('result/real_A_Y.csv', encoding='utf_8_sig')
real_B_Y.to_csv('result/real_B_Y.csv', encoding='utf_8_sig')
real_A_N.to_csv('result/real_A_N.csv', encoding='utf_8_sig')
real_B_N.to_csv('result/real_B_N.csv', encoding='utf_8_sig')
