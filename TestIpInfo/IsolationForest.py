import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from tqdm import tqdm
from datetime import datetime

# 将TCP标志位转换为数值向量
def tcp_flag_to_vector(tcp_flag):
    flag_mapping = {
        '-': 0,
        'A': 1,
        'B': 2,
        'C': 3,
        'D': 4,
        'E': 5,
        'F': 6,
        'G': 7,
        'H': 8,
        'I': 9,
        'J': 10,
        'K': 11,
        'L': 12,
        'M': 13,
        'N': 14,
        'O': 15,
        'P': 16,
        'Q': 17,
        'R': 18,
        'S': 19,
        'T': 20,
        'U': 21,
        'V': 22,
        'W': 23,
        'X': 24,
        'Y': 25,
        'Z': 26,
    }
    return [flag_mapping.get(char, -1) for char in tcp_flag]  # 使用 get 方法处理异常字符

def time_to_minutes(time_str):
    # 提取时间部分并转换为分钟数
    time_obj = datetime.strptime(time_str.split()[1], "%H:%M")
    minutes = time_obj.hour * 60 + time_obj.minute
    return minutes

def ip_to_numeric(ip_str):
    try:
        return [int(x) for x in ip_str.split('.')]
    except ValueError as e:
        print(f"Error converting IP address {ip_str}: {e}")
        return None
# 1.数据读取
data = pd.read_csv('data\\sample-data-202404\\sample-data-202404.csv')
label = pd.read_csv('data\\sample-data-202404\\sample-label-202404.csv')
# 2.合并数据集和标签
dataset = pd.concat([data, label], axis=1)
features = ['源IP', '目的IP', '源端口', '目的端口', 'TCP标志位', '开始时间', '包数', '字节数']
y = label['label']
# 3.将IP列转为四个整数的列
X = dataset[features].copy()
X.loc[:, '源IP_numeric'] = X['源IP'].apply(ip_to_numeric)
X.loc[:, '目的IP_numeric'] = X['目的IP'].apply(ip_to_numeric)
X = pd.concat([X.drop(['源IP_numeric', '目的IP_numeric'], axis=1),
               pd.DataFrame(X['源IP_numeric'].tolist()).add_prefix('源IP_'),
               pd.DataFrame(X['目的IP_numeric'].tolist()).add_prefix('目的IP_')], axis=1)
# 4.将开始时间转换为分钟数
X.loc[:, '开始时间'] = X['开始时间'].apply(lambda x: time_to_minutes(x))
# 5.将TCP标志位转换为数值向量并展开为多列
X.loc[:, 'TCP标志位'] = X['TCP标志位'].apply(lambda x: tcp_flag_to_vector(x))
X = pd.concat([X.drop(['TCP标志位'], axis=1), pd.DataFrame(X['TCP标志位'].tolist()).add_prefix('TCP标志位_')], axis=1)
# 6.更新features列表，添加新的IP相关特征（这里排除原始的TCP标志位、源IP和目的IP）
new_features = ['源端口', '目的端口', '开始时间', '包数', '字节数']
new_features.extend([col for col in X.columns if col.startswith('源IP_') or col.startswith('目的IP_') or col.startswith('TCP标志位_')])
features = new_features

X = X[features]
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 将训练集和测试集特征数据以及标签数据转换为numpy数组
X_train_array = X_train.values
X_test_array = X_test.values
y_train_array = y_train.values
y_test_array = y_test.values

# 模型训练(孤立森林)
# 设置训练轮次
num_epochs = 1
# 树的个数
n_estimators = 200
clf = IsolationForest(n_estimators=n_estimators, max_samples='auto', contamination=0.32, random_state=42)
batch_size = 1000
num_batches = int(np.ceil(len(X_train_array) / batch_size))

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}")
    for i in tqdm(range(num_batches)):
        start = i * batch_size
        end = (i + 1) * batch_size
        X_batch = X_train_array[start:end]
        y_batch = y_train_array[start:end]
        clf.fit(X_batch, y_batch)

# 模型评估
y_train_pred = clf.predict(X_train_array)
y_train_pred = [1 if x == -1 else 0 for x in y_train_pred]
y_test_pred = clf.predict(X_test_array)
y_test_pred = [1 if x == -1 else 0 for x in y_test_pred]
accuracy = accuracy_score(y_test_array, y_test_pred)
precision = precision_score(y_test_array, y_test_pred, zero_division=0)
recall = recall_score(y_test_array, y_test_pred)
f1 = f1_score(y_test_array, y_test_pred)
print("无监督的孤立森林")
print(f"Accuracy 准确率: {accuracy}")
print(f"Precision 精确率: {precision}")
print(f"Recall 召回率: {recall}")
print(f"F1 Score F1值: {f1}")

# 保存模型
joblib.dump(clf, 'model/isolated_forest_model.pkl')