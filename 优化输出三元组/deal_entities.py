import pandas as pd
import numpy as np

excel1 = pd.read_excel('protage.xlsx')

people_id = excel1.iloc[:3436, 0]
disease = excel1.iloc[:, 1]
drug = excel1.iloc[:416, 2]

combined_data = pd.concat([people_id, disease, drug], ignore_index=True)
entities = pd.DataFrame({'实体ID': range(len(combined_data)), '实体名称': combined_data})
entities['实体类型'] = np.where(entities['实体ID'] < len(people_id), '患者', np.where(entities['实体ID'] < len(people_id) + len(disease), '病症', '处方药'))

# 将结果保存到entities.csv中
entities.to_excel('entities.xlsx', index=False)
