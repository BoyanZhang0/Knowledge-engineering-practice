import pandas as pd
import os

# 读取CSV文件

# 数据表
df_A = pd.read_csv('Data/A.csv')
df_B = pd.read_csv('Data/B.csv')

# 知识图谱
KG_entities = pd.read_csv('Data/entities.csv')
KG_attr = pd.read_csv('Data/entities_attr.csv')
KG_relationships = pd.read_csv('Data/relationships.csv')






##Phase_1 - 样本数据处理##

### 抛弃特征 ###
print("### 抛弃特征 ###")
print("删去A表空值多于8成的列")
# 计算每一列的空值比例
missing_values = df_A.isnull().mean()
# 筛选空值比例大于八成的列
columns_to_drop = missing_values[missing_values > 0.8].index
# # 列出将要删除的属性
# print("从A表中删除属性：")
# print(columns_to_drop)
# 删除筛选出的列
df_A = df_A.drop(columns=columns_to_drop)
# 输出处理后的数据表信息
print("df_A.shape =", df_A.shape)  # 打印处理后数据表的形状
print("df_B.shape =", df_B.shape)

### 数据处理 ###
print("### 数据处理 ###")
df_A.drop("入院诊断 数值 ", axis=1, inplace=True)
df_B.drop("入院诊断 数值 ", axis=1, inplace=True)

# # 筛选出非int64和非float64的列
# non_numeric_columns = df_A.select_dtypes(exclude=['int64', 'float64'])
# # 输出非int64和非float64的列及其数据类型
# print("非int64和非float64的列及其数据类型：\n")
# for column in non_numeric_columns.columns:
#     print(f"列名: {column}, 数据类型: {non_numeric_columns[column].dtype}")

### 数据处理：数值转换 ###
print("### 数据处理：数值转换 ###")
# 定义一个函数，用于处理非数值元素
def data_process(element):
    replacements = {
        '阳性(+)': 1,
        '阳性（+）': 1,
        '阴性(-)': 0,
        '阴性（-）': 0,
        '阴性(－)': 0,
        '阴性（－）': 0,
        '弱阳性(±)': 0.5
    }
    if isinstance(element, str) and '<' in element:
        return float(element.replace('<', ''))
    elif isinstance(element, str) and '>' in element:
        return float(element.replace('>', ''))
    elif isinstance(element, str) and element in replacements:
        return replacements[element]
    else:
        return element

df_A = df_A.applymap(data_process)
df_B = df_B.applymap(data_process)
df_A = df_A.apply(pd.to_numeric, errors='ignore')
df_B = df_B.apply(pd.to_numeric, errors='ignore')


### 数据处理：空值填充 ###
# 查询相关资料，健康状态人体胰岛细胞抗体和抗胰岛素抗体(IAA) 均为阳性
# 这两组特征填充为1，其它填充为均值
print("### 数据处理：空值填充 ###")
# 定义一个函数，用于处理空值填充
def fill_missing_values(df):
    for column in df.columns:
        if column == "人体胰岛细胞抗体 数值" or column == "抗胰岛素抗体(IAA) 数值":
            df[column].fillna(1, inplace=True)
        else:
            # 尝试将非空值转换为数值类型，如果转换失败则跳过
            try:
                df[column] = pd.to_numeric(df[column], errors='coerce')
                mean_value = df[column][df[column].notnull()].mean()
                df[column].fillna(mean_value, inplace=True)
            except ValueError:
                pass
    return df
df_A = fill_missing_values(df_A)
df_B = fill_missing_values(df_B)

save_folder = 'Data\\processed\\'
# 检查路径是否存在，如果不存在则创建路径
if not os.path.exists(os.path.dirname(save_folder)):
    os.makedirs(os.path.dirname(save_folder))

# 将数据保存到指定路径
df_A.to_csv(save_folder + 'A_Processed.csv', index=False,encoding='utf-8-sig')
df_B.to_csv(save_folder + 'B_Processed.csv', index=False,encoding='utf-8-sig')
