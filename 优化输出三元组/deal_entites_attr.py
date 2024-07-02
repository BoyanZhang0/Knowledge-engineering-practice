import pandas as pd

excel1 = pd.read_excel('entities.xlsx')
excel2 = pd.read_excel('糖尿病病人住院数据.xlsx')
id = excel1.iloc[:3436, 0]

get_sex = excel2[excel2['id'].isin(excel1['实体ID'])]
entities_attr = pd.DataFrame({'实体ID': id})
entities_attr['实体属性'] = '性别'
entities_attr['实体属性值'] = get_sex.iloc[:, 3]
entities_attr.iloc[:, 2] = entities_attr.iloc[:, 2].astype(str) + '@int'
# 将结果保存到entities_attr.csv中
entities_attr.to_csv('entities_attr.csv', index=False)

