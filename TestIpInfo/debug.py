import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt

# 数据集相关列详细信息见 https://cloud.tencent.com/developer/article/1621977

# --------

# 数据读取
data_list = pd.read_csv("data/KDD_99/kddcup.data_10_percent_corrected.csv")


df = pd.DataFrame(data_list)
# 数据清洗(去除无label的记录)
# 第一列无意义 最后一列是标签 其他的都要作为输入特征处理
input_cols = df.columns[1:-1].tolist()
target_col = 'label'
# 筛选出类型为数值的列
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:-1]
# 数据预处理(用MinMaxScaler归一化数据，LabelEncoder将分类特征编码为数值)
scaler = MinMaxScaler()
scaler.fit(df[numeric_cols])
# 将缩放后的结果重新赋值给原数据框中对应的列 完成归一化处理
df[numeric_cols] = scaler.transform(df[numeric_cols])

le = LabelEncoder()
# 将label，protocol(协议类型),service(目标主机服务类型),flag(连接状态)列转化为数值
target = df['label']
# df['label'] = le.fit_transform(target)
df['label'] = [0 if x == 'normal.' else 1 for x in df['label']]
df['protocol_type'] = le.fit_transform(df['protocol_type'])
df['service'] = le.fit_transform(df['service'])
df['flag'] = le.fit_transform(df['flag'])


# 划分数据集 test=0.3
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)


train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

# 使用随机森林算法提取最重要的特征
rf = RandomForestClassifier(n_estimators=5, random_state=42)
rf.fit(train_inputs, train_targets)

# 获取每个特征的重要性得分
feature_importances = rf.feature_importances_

# 将特征重要性得分与特征名称对应起来
feature_importance_dict = dict(zip(train_inputs.columns, feature_importances))

# 按照重要性得分对特征进行排序（从高到低）
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# 选取前十个重要特征及其得分
top_ten_features = sorted_feature_importance[:10]

# 打印前十个特征及其重要性得分到控制台
for i, (feature, importance) in enumerate(top_ten_features):
    print(f"{i + 1}. Feature: {feature}, Importance Score: {importance}")