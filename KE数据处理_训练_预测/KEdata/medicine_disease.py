import numpy as np
import pandas as pd
import re

# 分别统计图谱中药和病的出现次数，并取前100种输出为csv文件。

# 读取文件
file_entities = 'KG/entities.csv'
file_relationship = 'KG/relationships.csv'

data_entities = pd.read_csv(file_entities, encoding='utf_8_sig')
data_relationship = pd.read_csv(file_relationship, encoding='utf_8_sig')

value_entities = data_entities.values[:, :]
value_relationship = data_relationship.values[:, :]

# 统计药和病
med = np.zeros(10000)
dis = np.zeros(10000)
for i in range(value_relationship.shape[0]):
    if value_relationship[i][2] == '出院诊断':
        dis[value_relationship[i][1]] += 1
    if value_relationship[i][2] == '出院带药':
        med[value_relationship[i][1]] += 1

# 排序取前100种
med_id = np.argsort(-med)
dis_id = np.argsort(-dis)

medicine = []
disease = []

for i in range(100):
    medicine.append([value_entities[med_id[i]][1], med_id[i]])
    disease.append([value_entities[dis_id[i]][1], dis_id[i]])

data_medicine = pd.DataFrame(medicine)
data_disease = pd.DataFrame(disease)

data_disease.to_csv('result/disease.csv', encoding='utf_8_sig')
data_medicine.to_csv('result/medicine.csv', encoding='utf_8_sig')
