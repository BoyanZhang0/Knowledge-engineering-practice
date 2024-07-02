import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 对数据进行评估并生成图表，文件名应符合格式:
# test_A_Y_ + 训练数 +.csv
# test_B_Y_ + 训练数 +.csv


# 导入测试数据结果并处理

file_real_A_Y = 'result/real_A_Y.csv'
file_real_A_N = 'result/real_A_N.csv'
file_real_B_Y = 'result/real_B_Y.csv'
file_real_B_N = 'result/real_B_N.csv'

real_A_Y = pd.read_csv(file_real_A_Y, encoding='utf_8_sig')
real_B_Y = pd.read_csv(file_real_B_Y, encoding='utf_8_sig')
real_A_N = pd.read_csv(file_real_A_N, encoding='utf_8_sig')
real_B_N = pd.read_csv(file_real_B_N, encoding='utf_8_sig')

real_A_Y = real_A_Y.values[:, 1:]
real_B_Y = real_B_Y.values[:, 1:]
real_A_N = real_A_N.values[:, 1]
real_B_N = real_B_N.values[:, 1]

real_A_N[real_A_N == 0] = 1
real_B_N[real_B_N == 0] = 1

def inv(i):
    file_test_A_Y = 'result/test_A_Y_'+str(i)+'.csv'
    file_test_B_Y = 'result/test_B_Y_'+str(i)+'.csv'

    test_A_Y = pd.read_csv(file_test_A_Y, encoding='utf_8_sig')
    test_B_Y = pd.read_csv(file_test_B_Y, encoding='utf_8_sig')

    test_A_Y = test_A_Y.values[:, 1:]
    test_B_Y = test_B_Y.values[:, 1:]

    Intersection_A = np.logical_and(real_A_Y, test_A_Y)
    Intersection_B = np.logical_and(real_B_Y, test_B_Y)

    Intersection_A = np.sum(Intersection_A, axis=1)
    Intersection_B = np.sum(Intersection_B, axis=1)
    test_A_N = np.sum(test_A_Y, axis=1)
    test_B_N = np.sum(test_B_Y, axis=1)

    test_A_N[test_A_N == 0] = 1
    test_B_N[test_B_N == 0] = 1

    Precision_A = Intersection_A / test_A_N
    Recall_A = Intersection_A / real_A_N
    x = Precision_A + Recall_A
    x[x == 0] = 1
    F1_A = 2 * Precision_A * Recall_A / x

    Precision_B = Intersection_B / test_B_N
    Recall_B = Intersection_B / real_B_N
    x = Precision_B + Recall_B
    x[x == 0] = 1
    F1_B = 2 * Precision_B * Recall_B / x

    Precision = np.append(Precision_A, Precision_B)
    Recall = np.append(Recall_A, Recall_B)
    F1 = np.append(F1_A, F1_B)

    # print('Precision_A ={}, Recall_A ={}, F1_A ={}'.format(Precision_A.mean(), Recall_A.mean(), F1_A.mean()))
    # print('Precision_B ={}, Recall_B ={}, F1_B ={}'.format(Precision_B.mean(), Recall_B.mean(), F1_B.mean()))
    # print('Precision ={}, Recall ={}, F1 ={}'.format(Precision.mean(), Recall.mean(), F1.mean()))
    return [[Precision_A.mean(), Recall_A.mean(), F1_A.mean()], [Precision_B.mean(), Recall_B.mean(), F1_B.mean()],
            [Precision.mean(), Recall.mean(), F1.mean()]]


PRF = []
PRF_A = []
PRF_B = []
X = []

# 对训练100——900轮的数据进行评估，步长100轮
for i in range(100, 901, 100):
    PRF_A.append(inv(i)[0])
    PRF_B.append(inv(i)[1])
    PRF.append(inv(i)[2])
    X.append(i)

# 对训练1000——10000轮的数据进行评估，步长1000轮
for i in range(1000, 10001, 1000):
    PRF_A.append(inv(i)[0])
    PRF_B.append(inv(i)[1])
    PRF.append(inv(i)[2])
    X.append(i)
PRF = np.array(PRF)

plt.plot(X, PRF[:, 0], label='Precision')
plt.plot(X, PRF[:, 1], label='Recall')
plt.plot(X, PRF[:, 2], label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.title('Result')
plt.savefig('result/result.png')
