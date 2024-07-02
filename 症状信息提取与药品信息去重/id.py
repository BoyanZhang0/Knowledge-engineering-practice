import pandas as pd


excel1 = pd.read_excel('糖尿病病人住院数据.xlsx')
excel2 = pd.read_excel('output.xlsx')
df = pd.DataFrame(columns=['id', '性别', '出院诊断', '用药情况'])
for index, row in excel2.iterrows():
    df.at[index, '用药情况'] = row.iloc[0]

# 遍历 excel_1 的每一行
for index, row in excel1.iterrows():
    if index == 0:
        continue
    if not pd.isnull(row.iloc[6]):
        df.at[index - 1, 'id'] = row.iloc[0]
        df.at[index - 1, '性别'] = row.iloc[3]
        df.at[index - 1, '出院诊断'] = row.iloc[5]

# 将结果保存到新的 Excel 文件
df.to_excel('test.xlsx', index=False)