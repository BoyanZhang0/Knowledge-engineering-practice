import numpy as np
import pandas as pd
import re

# 处理得到A组和B组训练输入和输出

# 读取数据
data_A = pd.read_csv('data/A_Processed.csv')
data_B = pd.read_csv('data/B_Processed.csv')
data_dis = pd.read_csv('result/disease.csv')
data_med = pd.read_csv('result/medicine.csv')
data_entities = pd.read_csv('KG/entities.csv')
data_relationships = pd.read_csv('KG/relationships.csv')

value_A = data_A.values
value_B = data_B.values
id_A = data_A.values[:, 1].tolist()
id_B = data_B.values[:, 1].tolist()
disease = data_dis.values[:, 2].tolist()
medicine = data_med.values[:, 2].tolist()
entities = data_entities.values
relationships = data_relationships.values

A_dis = np.zeros([value_A.shape[0], 100])
B_dis = np.zeros([value_B.shape[0], 100])
A_med = np.zeros([value_A.shape[0], 100])
B_med = np.zeros([value_B.shape[0], 100])


# 统计带药和患病情况
for i in range(relationships.shape[0]):
    x = int(float(entities[relationships[i][0]][1]))
    if x in id_A:
        if relationships[i][2] == '出院诊断' and relationships[i][1] in disease:
            A_dis[id_A.index(x)][disease.index(relationships[i][1])] = 1
        elif relationships[i][2] == '出院带药' and relationships[i][1] in medicine:
            A_med[id_A.index(x)][medicine.index(relationships[i][1])] = 1
    elif x in id_B:
        if relationships[i][2] == '出院诊断' and relationships[i][1] in disease:
            B_dis[id_B.index(x)][disease.index(relationships[i][1])] = 1
        elif relationships[i][2] == '出院带药' and relationships[i][1] in medicine:
            B_med[id_B.index(x)][medicine.index(relationships[i][1])] = 1
    else:
        print(relationships[i][0], "error")

# 得到训练矩阵和训练输出
train_A_X = pd.DataFrame(np.append(value_A, A_dis, axis=1))
train_B_X = pd.DataFrame(np.append(value_B, B_dis, axis=1))
train_A_Y = pd.DataFrame(A_med)
train_B_Y = pd.DataFrame(B_med)

train_A_X.to_csv('result/train_A_X.csv', index=False, columns=None, encoding='utf_8_sig')
train_B_X.to_csv('result/train_B_X.csv', index=False, columns=None, encoding='utf_8_sig')
train_A_Y.to_csv('result/train_A_Y.csv', columns=None, encoding='utf_8_sig')
train_B_Y.to_csv('result/train_B_Y.csv', columns=None, encoding='utf_8_sig')

