# 处理训练数据（包括输入和输出）和提取出现频率前100的药和病的名称
group.py负责将数据（这里的数据已经根据统计结果删去缺失值超过8成的4列）分为A、B两组

medicine_disease.py负责统计出现频率最多的100种药和病

processing_missing.py负责处理缺失值

train_x_y.py负责生成训练用的输入和输出
