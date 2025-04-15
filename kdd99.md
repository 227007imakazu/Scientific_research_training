## 使用KDDCUP99数据集训练和评估模型

#### 简介

**KDD Cup 1999数据集：** 

1998年美国国防部高级规划署（DARPA）在MIT林肯实验室进行了一项入侵检测评估项目。林肯实验室建立了模拟美国空军局域网的一个网络环境，收集了9周时间的TCPdump()网络连接和系统审计数据，仿真各种用户类型、各种不同的网络流量和攻击手段，使它就像一个真实的网络环境。这些TCPdump采集的原始数据被分为两个部分：7周时间的训练数据，大概包含5,000,000多个网络连接记录，剩下的2周时间的测试数据大概包含2,000,000个网络连接记录。

一个网络连接定义为在某个时间内从开始到结束的TCP数据包序列，并且在这段时间内，数据在预定义的协议下（如TCP、UDP）从源IP地址到目的IP地址的传递。每个网络连接被标记为正常（normal）或异常（attack），异常类型被细分为4大类共39种攻击类型，其中22种攻击类型出现在训练集中，另有17种未知攻击类型出现在测试集中。

4种异常类型分别是：

- DOS（denial-of-service）拒绝服务攻击，例如ping-of-death, syn flood, smurf等。

- R2L（unauthorized access from a remote machine to a local machine）来自远程主机的未授权访问，例如guessing password。

- U2R（unauthorized access to local superuser privileges by a local unpivileged user）未授权的本地超级用户特权访问，例如buffer overflow attacks。

- PROBING（surveillance and probing） 端口监视或扫描，例如port-scan, ping-sweep等。

随后来自哥伦比亚大学的Sal Stolfo 教授和来自北卡罗莱纳州立大学的 Wenke Lee 教授采用数据挖掘等技术对以上的数据集进行特征分析和数据预处理，形成了一个新的数据集。该数据集用于1999年举行的KDD CUP竞赛中，成为著名的KDD99数据集。虽然年代有些久远，但KDD99数据集仍然是网络入侵检测领域的事实Benckmark，为基于计算智能的网络入侵检测研究奠定基础。

#### 数据预处理

KDD99数据集大约有五百万条数据。在模型训练和评估中，考虑实际效率与时间因素，我采用了其10%数据集进行实验

```py
# 数据读取
data_list = pd.read_csv("data/KDD_99/kddcup.data_10_percent_corrected.csv")
df = pd.DataFrame(data_list)

# 数据清洗(去除无label的记录)
# 第一列无意义 最后一列是标签 其他的都要作为输入特征处理
input_cols = df.columns[1:-1].tolist()
target_col = 'label'

# 筛选出类型为数值的列
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:-1]

# 用MinMaxScaler归一化数据，LabelEncoder将分类特征编码为数值
scaler = MinMaxScaler()
scaler.fit(df[numeric_cols])

# 将缩放后的结果重新赋值给原数据框中对应的列 完成归一化处理
df[numeric_cols] = scaler.transform(df[numeric_cols])
le = LabelEncoder()

# 将label，protocol(协议类型),service(目标主机服务类型),flag(连接状态)列转化为数值
target = df['label']
# df['label'] = le.fit_transform(target)
# 这里由于kdd99数据集是一个多分类任务数据集,与我们的研究(异常检测)有出入，因此我将其中非正常的数据统一用0表示其为异常，1为正常
df['label'] = [1 if x == 'normal.' else 0 for x in df['label']]
df['protocol_type'] = le.fit_transform(df['protocol_type'])
df['service'] = le.fit_transform(df['service'])
df['flag'] = le.fit_transform(df['flag'])
```

#### 数据集划分与特征选取

因为该数据集共有41个输入参数(具体参数解释见[基于机器学习的入侵检测和攻击识别——以KDD CUP99数据集为例-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/1621977))，参数过多不仅训练时间长，而且容易造成模型过拟合，因此我采用随机森林选取其中最重要的若干个特征

```py
# 划分数据集 test=0.3
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

# 因为有41个输入特征 会导致模型过拟合 因此使用随机森林选取最重要的特征
sel = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
sel.fit(train_inputs, train_targets)
selected_feat = train_inputs.columns[(sel.get_support())]
print(selected_feat)
# 选取了11个特征参数
```

#### 模型训练与评估

```py
print("开始xun孤立森林算法模型...")
isolation_forest = IsolationForest(n_estimators=100, random_state=42, contamination=0.2)
isolation_forest.fit(train_inputs[selected_feat])
print("孤立森林算法模型训练完成...")
# 在训练集上预测异常分数（这里将其转化为分类结果的思路是：设定一个阈值，高于阈值为异常（=0），低于阈值为正常（=1））
# 因为这个数据集只有0.2的正常样本 用孤立森林操作不了 所以我将异常检测转为正常检测 让他去检测正常样本 所以normal=1，其余=0，

# 评估
test_preds_scores = isolation_forest.decision_function(test_inputs[selected_feat])
test_preds = np.where(test_preds_scores >= 0, 0, 1)

test_accuracy = accuracy_score(test_targets, test_preds)
test_precision = precision_score(test_targets, test_preds, average='macro')
test_recall = recall_score(test_targets, test_preds, average='macro')
test_f1 = f1_score(test_targets, test_preds, average='macro')

print("孤立森林")
print("准确率：", test_accuracy)
print("精确率：", test_precision)
print("召回率：", test_recall)
print("F1值：", test_f1)
```

目前 ，孤立森林模型在kdd9910%数据集的表现如下

![image-20241124123955287](E:\桌面\学校\科研训练\图片\孤立森林参数.png)

#### 完整代码

把其余模型的也放进去了，但是其他模型都很奇怪，准确率全是99..

```py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


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
# 这里由于kdd99数据集是一个多分类任务数据集,与我们的研究(异常检测)有出入，因此我将其中非正常的数据统一用0表示其为异常，1为正常
df['label'] = [1 if x == 'normal.' else 0 for x in df['label']]
df['protocol_type'] = le.fit_transform(df['protocol_type'])
df['service'] = le.fit_transform(df['service'])
df['flag'] = le.fit_transform(df['flag'])


# 划分数据集 test=0.3
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)


train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

# 因为有41个输入特征 会导致模型过拟合 因此使用随机森林选取最重要的特征
sel = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
sel.fit(train_inputs, train_targets)
selected_feat = train_inputs.columns[(sel.get_support())]
print(selected_feat)
# 选取了11个特征参数
# ---------------------------------------------------------------------------------------


# 模型训练

# 1. 随机森林
print("开始训练随机森林模型...")
rf = RandomForestClassifier(n_estimators=1000, random_state=42, verbose=1)
rf.fit(train_inputs[selected_feat], train_targets)
print("随机森林模型训练完成。")

preds_rf = rf.predict(test_inputs[selected_feat])
score_rf_accuracy = accuracy_score(test_targets, preds_rf)
score_rf_recall = recall_score(test_targets, preds_rf, average='macro')
score_rf_f1 = f1_score(test_targets, preds_rf, average='macro')
score_rf_precision = precision_score(test_targets, preds_rf, average='macro')

print("随机森林")
print("Accuracy of Random Forests 准确率: ", score_rf_accuracy)
print("Recall of Random Forests 召回率: ", score_rf_recall)
print("F1 Score of Random Forests F1值: ", score_rf_f1)
print("Precision of Random Forests 精确率: ", score_rf_precision)


# 2. 决策树
print("开始训练决策树模型...")
dc = DecisionTreeClassifier()
dc.fit(train_inputs[selected_feat], train_targets)
print("决策树模型训练完成。")

preds_dc = dc.predict(test_inputs[selected_feat])
score_dc_accuracy = accuracy_score(test_targets, preds_dc)
score_dc_recall = recall_score(test_targets, preds_dc, average='macro')
score_dc_f1 = f1_score(test_targets, preds_dc, average='macro')
score_dc_precision = precision_score(test_targets, preds_dc, average='macro')

print("决策树")
print("Accuracy of Decision Tree 准确率: ", score_dc_accuracy)
print("Recall of Decision Tree 召回率: ", score_dc_recall)
print("F1 Score of Decision Tree F1值: ", score_dc_f1)
print("Precision of Decision Tree 精确率: ", score_dc_precision)


# 3. k - 临近算法
print("开始训练K - 近邻算法模型...")
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(train_inputs[selected_feat], train_targets)
print("K - 近邻算法模型训练完成。")

preds_knn = knn.predict(test_inputs[selected_feat])
score_knn_accuracy = accuracy_score(test_targets, preds_knn)
score_knn_recall = recall_score(test_targets, preds_knn, average='macro')
score_knn_f1 = f1_score(test_targets, preds_knn, average='macro')
score_knn_precision = precision_score(test_targets, preds_knn, average='macro')

print("K-临近算法")
print("Accuracy of K Nearest Neighbors 准确率: ", score_knn_accuracy)
print("Recall of K Nearest Neighbors 召回率: ", score_knn_recall)
print("F1 Score of K Nearest Neighbors F1值: ", score_knn_f1)
print("Precision of K Nearest Neighbors 精确率: ", score_knn_precision)


# 4.孤立森林
print("开始训练孤立森林算法模型...")
isolation_forest = IsolationForest(n_estimators=100, random_state=42, contamination=0.2)
isolation_forest.fit(train_inputs[selected_feat])
print("孤立森林算法模型训练完成...")
# 在数据集上预测异常分数（这里将其转化为分类结果的思路是：设定一个阈值，高于阈值为异常（=0），低于阈值为正常（=1））
# 因为这个数据集只有0.2的正常样本 用孤立森林操作不了 所以我将异常检测转为正常检测 让他去检测正常样本 所以normal=1，其余=0，
# 评估
test_preds_scores = isolation_forest.decision_function(test_inputs[selected_feat])
test_preds = np.where(test_preds_scores >= 0, 0, 1)

test_accuracy = accuracy_score(test_targets, test_preds)
test_precision = precision_score(test_targets, test_preds, average='macro')
test_recall = recall_score(test_targets, test_preds, average='macro')
test_f1 = f1_score(test_targets, test_preds, average='macro')

print("孤立森林")
print("准确率：", test_accuracy)
print("精确率：", test_precision)
print("召回率：", test_recall)
print("F1值：", test_f1)

# 5.CNN
# 重新划分训练集和测试集，只使用选取的重要特征
train_df_selected = train_df[selected_feat.tolist() + [target_col]]
test_df_selected = test_df[selected_feat.tolist() + [target_col]]

train_inputs_selected = train_df_selected[selected_feat.tolist()].copy()
train_targets_selected = train_df_selected[target_col].copy()
test_inputs_selected = test_df_selected[selected_feat.tolist()].copy()
test_targets_selected = test_df_selected[target_col].copy()

# 构建神经网络模型（这里是一个简单的MLP）
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

# 训练模型
print("开始训练CNN模型...")
model.fit(train_inputs_selected, train_targets_selected)
print("CNN模型训练完成...")
# 在测试集上进行预测
test_predictions = model.predict(test_inputs_selected)

# 评估模型
accuracy = accuracy_score(test_targets_selected, test_predictions)
precision = precision_score(test_targets_selected, test_predictions)
recall = recall_score(test_targets_selected, test_predictions)
f1 = f1_score(test_targets_selected, test_predictions)
print("简单神经网络")
print("Accuracy 准确率:", accuracy)
print("Precision 精确率:", precision)
print("Recall: 召回率", recall)
print("F1 Score: F1值", f1)


# 优化:改变特征训练 修改参数等 后续再加
```

![image-20241124124244098](E:\桌面\学校\科研训练\图片\11.png)

![image-20241124124326024](E:\桌面\学校\科研训练\图片\12.png)
