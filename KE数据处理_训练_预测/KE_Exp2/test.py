# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
from model import FCNN
import argparse

import torch.nn.functional as F

import datetime
import os


# ##############################################################################
# Phase_0 - 命令行参数
################################################################################
print("Phase_0 - 命令行参数")

parser = argparse.ArgumentParser(description='Full-Connective NN of KE Practical Experiment')
parser.add_argument('--Group', type=str, default="GroupA",
                    help='group of data')
parser.add_argument('--data_X', type=str, default='Data/test_A_X.csv',
                    help='path to test feature matrix')
parser.add_argument('--model_path', type=str, default='Models/GroupA_5000ep_2024-04-02_19-48-26.pt',
                    help='path to trained model')
parser.add_argument('--cuda_able', action='store_true',
                    help='enables cuda')
parser.add_argument('--threshold', type=float, default=5.5,
                    help='threshold of classifying')

args = parser.parse_args()

print("\t参数设置完成\n")


# ##############################################################################
# Phase_1 - 加载测试数据
################################################################################
print("Phase_1 - 加载测试数据")

test_X = pd.read_csv(args.data_X)
test_data = np.array(test_X)[:, 2:]
sample_ids = np.array(test_X)[:, 1]

medicName = pd.read_csv('Data/medicName.csv')


# 转换为张量
test_X = torch.tensor(test_data, dtype=torch.float)

print("\t测试数据加载完成\n")

# ##############################################################################
# Phase_2 - 加载模型
################################################################################
print("Phase_2 - 加载模型")

fetlen = test_data.shape[1]
lablen = medicName.shape[0]

model = FCNN(fetlen, lablen, cuda=args.cuda_able)
model.load_state_dict(torch.load(args.model_path))

print("\t模型加载完成\n")

# ##############################################################################
# Phase_3 - 模型预测
# ##############################################################################
print("Phase_3 - 模型预测")

model.eval()
with torch.no_grad():
    outputs = model(test_X)



# print(predictions.shape)
# print(predictions)
# breakpoint()
# outputs = F.softmax(outputs, dim=1)
# test_result = outputs.numpy()

outputs = torch.sigmoid(outputs)
# 加入阈值处理
threshold = args.threshold
predictions = torch.gt(outputs, threshold).int()
test_result = predictions.numpy()

result_df = pd.DataFrame(test_result, columns=medicName['药名'])
# 将样本编号列插入到数据框的最前面
result_df.insert(0, '样本编号', sample_ids)

# 保存预测结果为Pandas数据框并存为表格
result_folder = 'testResult'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# 保存结果表格为带有BOM的UTF-8编码的CSV文件
result_filename = f'{args.Group}_test_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
result_path = os.path.join(result_folder, result_filename)

# 使用utf-8-sig编码保存CSV文件
result_df.to_csv(result_path, index=False, encoding='utf-8-sig')

print("\t预测完成")
print(f"\t预测结果已保存至 {result_path}")

