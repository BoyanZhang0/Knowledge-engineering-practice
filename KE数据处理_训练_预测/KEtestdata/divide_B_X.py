import pandas as pd
import re

data1 = pd.read_csv('data/test_B.csv')
data2 = pd.read_csv('data/medicine.csv')
# data0 = pd.read_csv('train_A_X.csv')
length, width = data1.shape

# 新增空列
for j in range(100):
    data1[83+j] = 0

# 列重命名
# list1 = [0, 0, 1, '出院诊断（先联）']
# list2 = list(range(181))
# del(list2[0:2])
# list1.extend(list2)
# data1.columns = list1

# 将病症赋到新列中
for i in range(length):
    j = 0
    for name in data2['0']:
        if re.search(name, data1['出院诊断（先联）'][i]) is not None:
            data1.iloc[i,j+83] = 1
        j += 1

data1 = data1.drop(columns=data1.columns[[3]])  # 删除汉字疾病列
# data1.drop("出院诊断（先联） ", axis=1, inplace=True)

def data_process(element):
    replacements = {
        '阳性(+)': 1,
        '阳性（+）': 1,
        '阴性(-)': 0,
        '阴性（-）': 0,
        '阴性(－)': 0,
        '阴性（－）': 0,
        '弱阳性(±)': 0.5,
        '可疑': 0.5
    }
    if isinstance(element, str) and '<' in element:
        return float(element.replace('<', ''))
    elif isinstance(element, str) and '>' in element:
        return float(element.replace('>', ''))
    elif isinstance(element, str) and element in replacements:
        return replacements[element]
    else:
        return element


df_B = data1.applymap(data_process)
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


df_B = fill_missing_values(df_B)

# 列重命名
list1 = [0, 0, 1]
list2 = list(range(181))
del(list2[0:2])
list1.extend(list2)
df_B.columns = list1

df_B.to_csv("result/test_B_X.csv", index=False, encoding='utf-8-sig')