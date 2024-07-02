import numpy as np
import pandas as pd
import re

# 将原数据根据中间缺失值的多少分为A、B两组，A组中间缺省值超过50%，B组缺省值小于等于50%；

# 读取文件
file_data = 'data/data.xlsx'
sheet_name = 'Sheet2'

data = pd.read_excel(file_data,sheet_name=sheet_name)

value = data.values[1:, :]
title = data.values[0, :]

A = []
B = []


# 根据缺失值比例进行划分
for i in range(value.shape[0]):
    n = 0
    for j in range(9, 69):
        if value[i][j] != value[i][j]:
            n += 1
    if n > 30:
        A.append(value[i])
    else:
        B.append(value[i])

group_A = pd.DataFrame(A,columns=title)
group_B = pd.DataFrame(B,columns=title)

group_A.to_csv('data/A.csv', encoding='utf_8_sig')
group_B.to_csv('data/B.csv', encoding='utf_8_sig')