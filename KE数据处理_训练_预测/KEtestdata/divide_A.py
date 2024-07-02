import pandas as pd
import re

data1 = pd.read_csv('data/test_A.csv')
data2 = pd.read_csv('data/medicine.csv')
# data0 = pd.read_csv('train_A_X.csv')
length, width = data1.shape

#  删除多余列
data1 = data1.drop(columns=data1.columns[[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                                         31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,
                                         51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69]])
# 新增空列
for j in range(100):
    data1[21+j] = 0
# 列重命名
data1 = data1.rename(columns={'id': '0', '性别 数值 ':'1', '入院体重指数 数值 ':'2', '入院收缩压':'3',
                              '院舒张压':'4', '入院腰围 数值 ':'5', '导出年龄':'6', '发病年龄':'7',
                              '出院带药缺失':'8', '妊娠':'9', '癌症':'10', '感染':'11', '糖尿病酮症':'12',
                              '糖尿病视网膜病变':'13', '糖尿病肾病':'14', '糖尿病周围神经病变':'15',
                              '下肢动脉病变':'16', '颈动脉病变':'17', '脑血管病':'18', '冠心病':'19', '高血压病':'20'})
# 将病症赋到新列中
for i in range(length):
    j = 0
    for name in data2['0']:
        if re.search(name, data1['出院诊断（先联）'][i]) is not None:
            data1.iloc[i,j+23] = 1
        j += 1

data1 = data1.drop(columns=data1.columns[[3]])  # 删除汉字疾病列
data1 = data1.fillna(value={'2':data1['2'].mean(), '3':data1['3'].mean(), '4':data1['4'].mean(),  # 缺省值补全
                            '5':data1['5'].mean(), '6':data1['6'].mean(), '7':data1['7'].mean()})

print(data1)
data1.to_csv("result/test_A_X.csv", index=False)