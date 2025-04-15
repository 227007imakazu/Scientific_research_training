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
sel = SelectFromModel(RandomForestClassifier(n_estimators=5, random_state=42)) # 原是100
sel.fit(train_inputs, train_targets)
selected_feat = train_inputs.columns[(sel.get_support())]
print(selected_feat)
# 选取了11个特征参数
# -------------------------------------------------------------------------------------------------------


# 模型训练

# 1. 随机森林
# print("开始训练随机森林模型...")
# rf = RandomForestClassifier(n_estimators=1000, random_state=42, verbose=1)
# rf.fit(train_inputs[selected_feat], train_targets)
# print("随机森林模型训练完成。")
#
# preds_rf = rf.predict(test_inputs[selected_feat])
# score_rf_accuracy = accuracy_score(test_targets, preds_rf)
# score_rf_recall = recall_score(test_targets, preds_rf, average='macro')
# score_rf_f1 = f1_score(test_targets, preds_rf, average='macro')
# score_rf_precision = precision_score(test_targets, preds_rf, average='macro')
#
# print("随机森林")
# print("Accuracy of Random Forests 准确率: ", score_rf_accuracy)
# print("Recall of Random Forests 召回率: ", score_rf_recall)
# print("F1 Score of Random Forests F1值: ", score_rf_f1)
# print("Precision of Random Forests 精确率: ", score_rf_precision)


# 2. 决策树
# print("开始训练决策树模型...")
# dc = DecisionTreeClassifier()
# dc.fit(train_inputs[selected_feat], train_targets)
# print("决策树模型训练完成。")
#
# preds_dc = dc.predict(test_inputs[selected_feat])
# score_dc_accuracy = accuracy_score(test_targets, preds_dc)
# score_dc_recall = recall_score(test_targets, preds_dc, average='macro')
# score_dc_f1 = f1_score(test_targets, preds_dc, average='macro')
# score_dc_precision = precision_score(test_targets, preds_dc, average='macro')
#
# print("决策树")
# print("Accuracy of Decision Tree 准确率: ", score_dc_accuracy)
# print("Recall of Decision Tree 召回率: ", score_dc_recall)
# print("F1 Score of Decision Tree F1值: ", score_dc_f1)
# print("Precision of Decision Tree 精确率: ", score_dc_precision)


# 3. k - 临近算法
# print("开始训练K - 近邻算法模型...")
# knn = KNeighborsClassifier(n_neighbors=7)
# knn.fit(train_inputs[selected_feat], train_targets)
# print("K - 近邻算法模型训练完成。")
#
# preds_knn = knn.predict(test_inputs[selected_feat])
# score_knn_accuracy = accuracy_score(test_targets, preds_knn)
# score_knn_recall = recall_score(test_targets, preds_knn, average='macro')
# score_knn_f1 = f1_score(test_targets, preds_knn, average='macro')
# score_knn_precision = precision_score(test_targets, preds_knn, average='macro')
#
# print("K-临近算法")
# print("Accuracy of K Nearest Neighbors 准确率: ", score_knn_accuracy)
# print("Recall of K Nearest Neighbors 召回率: ", score_knn_recall)
# print("F1 Score of K Nearest Neighbors F1值: ", score_knn_f1)
# print("Precision of K Nearest Neighbors 精确率: ", score_knn_precision)


# 4.孤立森林
# print("开始训练孤立森林算法模型...")
# isolation_forest = IsolationForest(n_estimators=100, random_state=42, contamination=0.2)
# isolation_forest.fit(train_inputs[selected_feat])
# print("孤立森林算法模型训练完成...")
# # 在训练集上预测异常分数（这里将其转化为分类结果的思路是：设定一个阈值，高于阈值为异常（=0），低于阈值为正常（=1））
# # 因为这个数据集只有0.2的正常样本 用孤立森林操作不了 所以我将异常检测转为正常检测 让他去检测正常样本 所以normal=1，其余=0，
#
# # 评估
# test_preds_scores = isolation_forest.decision_function(test_inputs[selected_feat])
# test_preds = np.where(test_preds_scores >= 0, 0, 1)
#
# test_accuracy = accuracy_score(test_targets, test_preds)
# test_precision = precision_score(test_targets, test_preds, average='macro')
# test_recall = recall_score(test_targets, test_preds, average='macro')
# test_f1 = f1_score(test_targets, test_preds, average='macro')
#
# print("孤立森林")
# print("准确率：", test_accuracy)
# print("精确率：", test_precision)
# print("召回率：", test_recall)
# print("F1值：", test_f1)

# 5.CNN
# 重新划分训练集和测试集，只使用选取的重要特征
# train_df_selected = train_df[selected_feat.tolist() + [target_col]]
# test_df_selected = test_df[selected_feat.tolist() + [target_col]]
#
# train_inputs_selected = train_df_selected[selected_feat.tolist()].copy()
# train_targets_selected = train_df_selected[target_col].copy()
# test_inputs_selected = test_df_selected[selected_feat.tolist()].copy()
# test_targets_selected = test_df_selected[target_col].copy()
#
# # 构建神经网络模型（这里是一个简单的MLP）
# model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
#
# # 训练模型
# print("开始训练CNN模型...")
# model.fit(train_inputs_selected, train_targets_selected)
# print("CNN模型训练完成...")
# # 在测试集上进行预测
# test_predictions = model.predict(test_inputs_selected)
#
# # 评估模型
# accuracy = accuracy_score(test_targets_selected, test_predictions)
# precision = precision_score(test_targets_selected, test_predictions)
# recall = recall_score(test_targets_selected, test_predictions)
# f1 = f1_score(test_targets_selected, test_predictions)
# print("简单神经网络")
# print("Accuracy 准确率:", accuracy)
# print("Precision 精确率:", precision)
# print("Recall: 召回率", recall)
# print("F1 Score: F1值", f1)





# 优化:改变特征训练 修改参数等 后续再加


# rf = RandomForestClassifier(n_estimators=1000, random_state=42)
# rf.fit(train_inputs[selected_feat], train_targets);
# preds_rf = rf.predict(test_inputs[selected_feat])
# score_rf = accuracy_score(test_targets, preds_rf)
# print("Accuracy of Random Forests: ", score_rf)
#
# dc = DecisionTreeClassifier()
# dc.fit(train_inputs[selected_feat], train_targets);
# preds_dc = dc.predict(test_inputs[selected_feat])
# score_dc = accuracy_score(test_targets, preds_dc)
# print("Accuracy of Decision Tree: ", score_dc)
#
# knn = KNeighborsClassifier(n_neighbors=7)
# knn.fit(train_inputs[selected_feat], train_targets)
# preds_knn = knn.predict(test_inputs[selected_feat])
# score_knn = accuracy_score(test_targets, preds_knn)
# print("Accuracy of K Nearest Neighbors: ", score_knn)
#
# rf_1 = RandomForestClassifier(n_estimators=1000, random_state=42)
# rf_1.fit(train_inputs[["diff_srv_rate", "same_srv_rate", "dst_host_same_srv_rate", "flag",
#                        "count", "dst_host_srv_count", "dst_host_srv_serror_rate"]], train_targets);
# preds_rf_1 = rf_1.predict(test_inputs[["diff_srv_rate", "same_srv_rate",
#                                        "dst_host_same_srv_rate", "flag", "count", "dst_host_srv_count",
#                                        "dst_host_srv_serror_rate"]])
# score_rf_1 = accuracy_score(test_targets, preds_rf_1)
# print("Accuracy of Random Forest Classifier is: ", score_rf_1)
#
# dc_1 = DecisionTreeClassifier()
# dc_1.fit(train_inputs[["diff_srv_rate", "same_srv_rate", "dst_host_same_srv_rate",
#                        "flag", "count", "dst_host_srv_count", "dst_host_srv_serror_rate"]], train_targets);
# preds_dc_1 = dc_1.predict(test_inputs[["diff_srv_rate", "same_srv_rate",
#                                        "dst_host_same_srv_rate", "flag", "count", "dst_host_srv_count",
#                                        "dst_host_srv_serror_rate"]])
# score_dc_1 = accuracy_score(test_targets, preds_rf_1)
# print("Accuracy of Decision Tree Classifier is :",score_dc_1)
#
# knn_1 = KNeighborsClassifier(n_neighbors=7)
# knn_1.fit(train_inputs[["diff_srv_rate", "same_srv_rate",
#                         "dst_host_same_srv_rate", "flag", "count", "dst_host_srv_count", "dst_host_srv_serror_rate"]],
#           train_targets)
# preds_knn_1 = knn_1.predict(test_inputs[["diff_srv_rate",
#                                          "same_srv_rate", "dst_host_same_srv_rate", "flag", "count",
#                                          "dst_host_srv_count", "dst_host_srv_serror_rate"]])
# score_knn_1 = accuracy_score(test_targets, preds_knn_1)
# print("Accuracy of K Nearest Neighbors is: ", score_knn_1)
