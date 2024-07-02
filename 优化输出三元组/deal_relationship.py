import pandas as pd
import numpy as np

df = pd.read_excel('merge.xlsx')
excel1 = df[df.iloc[:, 2] != '性别']
excel2 = pd.read_excel('entities.xlsx')

index_values = excel2.iloc[:, 1].tolist()
data_values = excel2.iloc[:, 0].tolist()

# 构建字典
dictionary = dict(zip(index_values, data_values))
relationships = pd.DataFrame({'头实体ID': excel1['头实体ID'].map(dictionary), '尾实体ID': excel1['尾实体ID'].map(dictionary), '实体关系': excel1['实体关系']})
# 将结果保存到entities.csv中
relationships.to_csv('relationships.csv', index=False)
