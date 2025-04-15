# 模型评估
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 读取结果文件和标签文件
result_df = pd.read_csv('data/sample-data-202404/output/sample_results_202404.csv')
label_df = pd.read_csv('data/sample-data-202404/sample-label-202404.csv')

# 获取检测结果列和真实标签列
y_pred = result_df['result']
y_true = label_df['label']

# 确保结果和标签的行数相同，如果不一致，可能需要进行数据清洗或调整
if len(y_pred)!= len(y_true):
    raise ValueError("检测结果和真实标签的行数不一致，请检查数据！")

# 计算准确率
accuracy = accuracy_score(y_true, y_pred)

# 计算召回率
recall = recall_score(y_true, y_pred)

# 计算精确率
precision = precision_score(y_true, y_pred)

# 计算F1值
f1 = f1_score(y_true, y_pred)

print("准确率:", accuracy)
print("召回率:", recall)
print("精确率:", precision)
print("F1值:", f1)