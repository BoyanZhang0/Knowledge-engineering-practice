'''import torch
import numpy as np
import pandas as pd

def get_token(input):
    # english = 'abcdefghijklmnopqrstuvwxyz0123456789'
    english = ''
    output = []
    buffer = ''
    for s in input:
        if s in english or s in english.upper():
            buffer += s
        else:
            if buffer: output.append(buffer)
            buffer = ''
            output.append(s)
    if buffer: output.append(buffer)
    return output


from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained('./checkpoint/model/bert-base-chinese-15epoch')
print(model)
print(model.config.id2label)

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')


if __name__ == '__main__':
    input_str = '百泌达10ugbid(每天两次,每次10ug,餐前30分钟)、安达唐10mgtid(每天一次,每次一片,餐前)、格华止1.0gbid(每天两次,每次两片,餐前)、以上为降糖方案,根据血糖调整患者胰岛功能差,严密监测血糖变化安博维150mgqd(每日一次,每次一片,晨起)、以上为降压药物,根据血压调整药物剂量,心脏科随诊立普妥20mgqn(每晚一次,每次一片)、以上为降脂药物,一月后复查肝功能及血脂甲钴胺0.5mgtid(每日三次,每次一片)、以上为营养神经药物,可长期服用,内分泌科门诊随诊怡开120iutid(每天三次,每次一片)、以上为改善微循环药物,可长期服用,内分泌科门诊随诊阿法骨化三醇0.5ugqd(每天一次,每次两粒)、以上为补充维生素D药物,可长期服用,内分泌科随诊维固力0.5ugqd(每天一次,每次两粒)、以上为保护关节药物,可长期服用,骨科随诊迈之灵150mgbid(每天两次,每次一粒)、以上为改善关节肿痛药物,骨科随诊安必丁50mgqd(每天一次,每次一粒)、'
    input_char = get_token(input_str)
    input_tensor = tokenizer(input_char, is_split_into_words=True, padding=True, truncation=True,
                             return_offsets_mapping=True, max_length=512, return_tensors="pt")
    input_tokens = input_tensor.tokens()
    offsets = input_tensor["offset_mapping"]
    ignore_mask = offsets[0, :, 1] == 0
    # print(input_tensor)
    input_tensor.pop("offset_mapping")  # 不剔除的话会报错
    outputs = model(**input_tensor)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
    predictions = outputs.logits.argmax(dim=-1)[0].tolist()
    print(predictions)
    results = []

    tokens = input_tensor.tokens()
    idx = 0
    while idx < len(predictions):
        if ignore_mask[idx]:
            idx += 1
            continue
        pred = predictions[idx]
        label = model.config.id2label[pred]
        if label != "O":
            # 不加B-或者I-
            label = label[2:]
            start = idx
            end = start + 1
            # 获取所有token I-label
            all_scores = []
            all_scores.append(probabilities[start][predictions[start]])
            while (
                    end < len(predictions)
                    and model.config.id2label[predictions[end]] == f"I-{label}"
            ):
                all_scores.append(probabilities[end][predictions[end]])
                end += 1
                idx += 1
            # 得到是他们平均的
            score = np.mean(all_scores).item()
            word = input_tokens[start:end]
            results.append(
                {
                    "entity_group": label,
                    "score": score,
                    "word": "".join(word),
                    "start": start,
                    "end": end,
                }
            )
        idx += 1

    for i in range(len(results)):
        print(results[i])

    #print(results)'''
        
import torch
import numpy as np
import pandas as pd
import os
from openpyxl import load_workbook

def get_token(input):
    # english = 'abcdefghijklmnopqrstuvwxyz0123456789'
    english = ''
    output = []
    buffer = ''
    for s in input:
        if s in english or s in english.upper():
            buffer += s
        else:
            if buffer: output.append(buffer)
            buffer = ''
            output.append(s)
    if buffer: output.append(buffer)
    return output

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained('./checkpoint/model/bert-base-chinese-15epoch')
print(model)
print(model.config.id2label)

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
if __name__ == '__main__':
    input_str = []
    file_path = "C:\\Users\\25494\\Desktop\\ner\\糖尿病病人住院数据.xlsx"  # 替换为您的文本文件夹路径
    sheet_name = "Sheet2"  # 假设数据在名为"Sheet1"的工作表中
    column_index = 6  # 第七列的索引为6（索引从0开始）
    data = pd.read_excel(file_path, sheet_name=sheet_name)# 读取Excel文件中的数据
    column_data = data.iloc[3:3607, column_index].astype(str)# 提取第七列的文本数据
    input_str = column_data.tolist()# 将文本数据添加到input_str列表中
    results = []
    for str_1 in input_str:
        input_char = get_token(str_1)
        input_tensor = tokenizer(input_char, is_split_into_words=True, padding=True, truncation=True,
                                return_offsets_mapping=True, max_length=512, return_tensors="pt")
        input_tokens = input_tensor.tokens()
        offsets = input_tensor["offset_mapping"]
        ignore_mask = offsets[0, :, 1] == 0
        # print(input_tensor)
        input_tensor.pop("offset_mapping")  # 不剔除的话会报错
        outputs = model(**input_tensor)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
        predictions = outputs.logits.argmax(dim=-1)[0].tolist()
        #print(predictions)
        tokens = input_tensor.tokens()
        idx = 0
        temp_dict = {}
        while idx < len(predictions):
            if ignore_mask[idx]:
                idx += 1
                continue
            pred = predictions[idx]
            label = model.config.id2label[pred]
            if label != "O":
                # 不加B-或者I-
                label = label[2:]
                start = idx
                end = start + 1
                # 获取所有token I-label
                all_scores = []
                all_scores.append(probabilities[start][predictions[start]])
                while (
                        end < len(predictions)
                        and model.config.id2label[predictions[end]] == f"I-{label}"
                ):
                    all_scores.append(probabilities[end][predictions[end]])
                    end += 1
                    idx += 1
                # 得到是他们平均的
                score = np.mean(all_scores).item()
                word = input_tokens[start:end]
                #print(word)
                # 结果添加到临时字典中
                if label not in temp_dict:
                    temp_dict[label] = {
                        "entity_group": label,
                        "score": [],
                        "word": [],
                        "start": [],
                        "end": []
                    }
                temp_dict[label]["score"].append(score)
                temp_dict[label]["word"].append("".join(word))
                temp_dict[label]["start"].append(start)
                temp_dict[label]["end"].append(end)

            idx += 1

        # 将临时字典中的结果添加到结果列表中
        for label, entity_dict in temp_dict.items():
            results.append(entity_dict)

    # 创建数据框
    df = pd.DataFrame(results)

    #创建Excel表格
    excel_data = []
    for index, row in df.iterrows():
        words = row["word"]
        excel_data.append({"药品名称": ", ".join(words)})

    excel_df = pd.DataFrame(excel_data)

    #保存为Excel文件
    excel_df.to_excel("./output.xlsx", index=False)
    #for i in range(len(results)):
        #print(results[i])
    #print(results)