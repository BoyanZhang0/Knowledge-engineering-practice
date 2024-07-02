# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from model import FCNN
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import datetime
import argparse

# ##############################################################################
# Phase_0 - 命令行参数
################################################################################
print("Phase_0 - 命令行参数")

parser = argparse.ArgumentParser(description='Full-Connective NN of KE Practical Experiment')
parser.add_argument('--Group', type=str, default="GroupA",
                    help='group of data')
parser.add_argument('--data_X', type=str, default='Data/train_A_X.csv',
                    help='path to feature matrix')
parser.add_argument('--data_Y', type=str, default='Data/train_A_Y.csv',
                    help='path to label matrix')
parser.add_argument('--epochs', type=int, default=2000,
                    help='number of epochs for train')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--validate_every', type=int, default=20,
                    help='amount of epochs of each validation')
parser.add_argument('--redivide_every', type=int, default=2,
                    help='amount of validation of each datasets-redivide')

parser.add_argument('--cuda_able', action='store_true',
                    help='enables cuda')

parser.add_argument('--pretrain', action='store_true',
                    help='load trained models')
parser.add_argument('--preModel', type=str, default='Models/GroupA_2000ep_2024-04-02_16-30-43.pt',
                    help='path to trained models')
parser.add_argument('--threshold', type=float, default=0.6,
                    help='threshold of classifying')


args = parser.parse_args()

print("\t参数设置完成\n")
# print(args)
# breakpoint()

# ##############################################################################
# Phase_1 - 数据加载
################################################################################
print("Phase_1 - 数据加载")
# A组数据导入
# parser here


train_X = pd.read_csv(args.data_X)
train_Y = pd.read_csv(args.data_Y)
# 切片保留特征和标签矩阵
data = np.array(train_X)[:, 2:]
labels = np.array(train_Y)[:, 1:]

timestamp = int(datetime.datetime.now().timestamp())
train_X, val_X, train_Y, val_Y = train_test_split(data, labels, test_size=0.2, random_state=timestamp)

# 转换为张量
train_X = torch.tensor(train_X, dtype=torch.float)
train_Y = torch.tensor(train_Y, dtype=torch.float)
val_X = torch.tensor(val_X, dtype=torch.float)
val_Y = torch.tensor(val_Y, dtype=torch.float)

print("\t数据加载完成\n")

# ##############################################################################
# Phase_2 - 初始化
################################################################################
print("Phase_2 - 初始化")
fetlen = data.shape[1]
lablen = labels.shape[1]

# 初始化模型、损失函数和优化器
model = FCNN(fetlen, lablen, cuda=args.cuda_able)

# 交叉熵损失通常用于多分类问题
# criterion = nn.CrossEntropyLoss()
# 此处为多组二分类问题，采用二进制交叉熵损失
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=args.lr)

print("\t初始化完成")

# 指定之前保存的模型文件路径

# parser here
ifPre = args.pretrain
if ifPre:
    # parser here
    model_filepath = args.preModel
    # 加载模型状态字典
    loaded_model_state = torch.load(model_filepath)
    # 将加载的模型权重应用于当前模型
    model.load_state_dict(torch.load(model_filepath))
    print("\t已加载并应用保存的模型权重到当前模型。")
print("\n")

# ##############################################################################
# Phase_3 - 模型训练
################################################################################
# 当前时间
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 创建一个列表来存储验证损失
val_losses = []
precisions = []
recalls = []
f1_scores = []


# 训练函
def train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs, validate_every,
                redivide_every,file_path,threshold):
    with open(file_path, 'w') as file:
        file.write(f'Training Log - {current_time}\n')
    # 训练循环
        for epoch in tqdm(range(num_epochs), desc='Training Progress'):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train.squeeze().float())
            loss.backward()
            optimizer.step()

            # 验证循环
            if (epoch + 1) % validate_every == 0:
                val_loss, val_precision, val_recall, val_f1 = validate(model, criterion, X_val, y_val,threshold)
                # print(f'| Epoch {epoch + 1}/{num_epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} |')
                file.write(f'| Epoch {epoch + 1}/{num_epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1 Score: {val_f1:.4f} |\n')
                val_losses.append(val_loss)
                precisions.append(val_precision)
                recalls.append(val_recall)
                f1_scores.append(val_f1)

                # 动态划分验证集
                if (epoch + 1) % (validate_every * redivide_every) == 0:
                    combined_X = torch.cat((X_train, X_val), 0)
                    combined_y = torch.cat((y_train, y_val), 0)
                    combined_X = combined_X.numpy()
                    combined_y = combined_y.numpy()

                    timestamp = int(datetime.datetime.now().timestamp())
                    new_train_X, new_val_X, new_train_Y, new_val_Y = train_test_split(combined_X, combined_y, test_size=0.2,
                                                                                      random_state=timestamp)
                    X_train = torch.from_numpy(new_train_X)
                    X_val = torch.from_numpy(new_val_X)
                    y_train = torch.from_numpy(new_train_Y)
                    y_val = torch.from_numpy(new_val_Y)


# 验证函
def validate(model, criterion, X_val, y_val, threshold):
    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        loss = criterion(outputs, y_val.squeeze().float())

        # 根据阈值确定预测选择的药品
        predicted_labels = torch.gt(outputs, threshold).int()

        # 计算准确率、召回率和F1分数
        true_positives = torch.sum(predicted_labels * y_val).item()
        predicted_positives = torch.sum(predicted_labels).item()
        actual_positives = torch.sum(y_val).item()

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return loss.item(), precision, recall, f1


# 训练和验证

# parser here
num_epochs = args.epochs
validate_every = args.validate_every
redivide_every = args.redivide_every

# 保存训练记录
if not os.path.exists('Log'):
    os.makedirs('Log')
# 定义文件名
file_name = f'{args.Group}_{num_epochs}_{current_time}.txt'
file_path = os.path.join("Log", file_name)

# 训练A组模型
train_model(model, criterion, optimizer, train_X, train_Y, val_X, val_Y, num_epochs, validate_every, redivide_every,file_path,args.threshold)

print(f'Training log saved at: {file_path}')


# ##############################################################################
# Phase_4 - 结果可视化
# ##############################################################################

plt.figure(figsize=(12, 6))

# 绘制验证损失
plt.subplot(1, 2, 1)
plt.plot(range(0, num_epochs, validate_every), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Validation Loss of {args.Group}')
plt.legend()

# 绘制准确率、召回率和F1分数
plt.subplot(1, 2, 2)
plt.plot(range(0, num_epochs, validate_every), precisions, label='Precision')
plt.plot(range(0, num_epochs, validate_every), recalls, label='Recall')
plt.plot(range(0, num_epochs, validate_every), f1_scores, label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title(f'Metrics of {args.Group}')
plt.legend()

# 添加epoch数和日期到图像
plt.text(num_epochs - 5, 0.9, f'lr: {args.lr}', ha='right',zorder=10)
plt.text(num_epochs - 5, 0.85, f'Epochs: {num_epochs}', ha='right',zorder=10)
plt.text(num_epochs - 5, 0.80, f'Date: {current_time}', ha='right',zorder=10)

# 保存图像到Fig文件夹
if not os.path.exists('Fig'):
    os.makedirs('Fig')

file_name = f'{args.Group}_{num_epochs}ep_{current_time}.png'
plt.savefig(f'Fig/{file_name}')
plt.show()

# ##############################################################################
# Phase_5 - 模型保存
# ###############################################################################

# 模型会保存到根目录文件夹Models

if not os.path.exists('Models'):
    os.makedirs('Models')

folder_path = "Models"
model_filename = f"{args.Group}_{num_epochs}ep_{current_time}.pt"
model_filepath = os.path.join(folder_path, model_filename)

# 保存模型状态字典到文件
torch.save(model.state_dict(), model_filepath)

print(f"模型已保存到文件: {model_filepath}")