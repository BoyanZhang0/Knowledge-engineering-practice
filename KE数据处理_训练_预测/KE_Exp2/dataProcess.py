import pandas as pd
import os

# ��ȡCSV�ļ�

# ���ݱ�
df_A = pd.read_csv('Data/A.csv')
df_B = pd.read_csv('Data/B.csv')

# ֪ʶͼ��
KG_entities = pd.read_csv('Data/entities.csv')
KG_attr = pd.read_csv('Data/entities_attr.csv')
KG_relationships = pd.read_csv('Data/relationships.csv')






##Phase_1 - �������ݴ���##

### �������� ###
print("### �������� ###")
print("ɾȥA���ֵ����8�ɵ���")
# ����ÿһ�еĿ�ֵ����
missing_values = df_A.isnull().mean()
# ɸѡ��ֵ�������ڰ˳ɵ���
columns_to_drop = missing_values[missing_values > 0.8].index
# # �г���Ҫɾ��������
# print("��A����ɾ�����ԣ�")
# print(columns_to_drop)
# ɾ��ɸѡ������
df_A = df_A.drop(columns=columns_to_drop)
# ������������ݱ���Ϣ
print("df_A.shape =", df_A.shape)  # ��ӡ��������ݱ����״
print("df_B.shape =", df_B.shape)

### ���ݴ��� ###
print("### ���ݴ��� ###")
df_A.drop("��Ժ��� ��ֵ ", axis=1, inplace=True)
df_B.drop("��Ժ��� ��ֵ ", axis=1, inplace=True)

# # ɸѡ����int64�ͷ�float64����
# non_numeric_columns = df_A.select_dtypes(exclude=['int64', 'float64'])
# # �����int64�ͷ�float64���м�����������
# print("��int64�ͷ�float64���м����������ͣ�\n")
# for column in non_numeric_columns.columns:
#     print(f"����: {column}, ��������: {non_numeric_columns[column].dtype}")

### ���ݴ�����ֵת�� ###
print("### ���ݴ�����ֵת�� ###")
# ����һ�����������ڴ������ֵԪ��
def data_process(element):
    replacements = {
        '����(+)': 1,
        '���ԣ�+��': 1,
        '����(-)': 0,
        '���ԣ�-��': 0,
        '����(��)': 0,
        '���ԣ�����': 0,
        '������(��)': 0.5
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


### ���ݴ�����ֵ��� ###
# ��ѯ������ϣ�����״̬�����ȵ�ϸ������Ϳ��ȵ��ؿ���(IAA) ��Ϊ����
# �������������Ϊ1���������Ϊ��ֵ
print("### ���ݴ�����ֵ��� ###")
# ����һ�����������ڴ����ֵ���
def fill_missing_values(df):
    for column in df.columns:
        if column == "�����ȵ�ϸ������ ��ֵ" or column == "���ȵ��ؿ���(IAA) ��ֵ":
            df[column].fillna(1, inplace=True)
        else:
            # ���Խ��ǿ�ֵת��Ϊ��ֵ���ͣ����ת��ʧ��������
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
# ���·���Ƿ���ڣ�����������򴴽�·��
if not os.path.exists(os.path.dirname(save_folder)):
    os.makedirs(os.path.dirname(save_folder))

# �����ݱ��浽ָ��·��
df_A.to_csv(save_folder + 'A_Processed.csv', index=False,encoding='utf-8-sig')
df_B.to_csv(save_folder + 'B_Processed.csv', index=False,encoding='utf-8-sig')
