import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import joblib

# 将TCP标志位转换为数值向量
def tcp_flag_to_vector(tcp_flag):
    # 定义TCP标志位的映射关系
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

# 将开始时间转换为分钟数
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
# 数据读取
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

# 处理数据类型，确保可以转换为张量并移动到GPU
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].astype(str).astype(float)
    if X_test[col].dtype == 'object':
        X_test[col] = X_test[col].astype(str).astype(float)

# 转换为numpy数组并移动到GPU
X_train = torch.tensor(X_train.values, dtype=torch.float32).cuda()
X_test = torch.tensor(X_test.values, dtype=torch.float32).cuda()
y_train = torch.tensor(y_train.values, dtype=torch.float32).cuda()
y_test = torch.tensor(y_test.values, dtype=torch.float32).cuda()

# 模型训练
# 1.定义简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out

# 初始化模型、损失函数和优化器
model = SimpleNN(X_train.shape[1]).cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', patience=3, factor=0.1)
num_epochs = 200  # 轮数
batch_size = 10000  # 批量大小
num_batches = len(X_train) // batch_size + (1 if len(X_train) % batch_size != 0 else 0)

with tqdm(total=num_epochs, desc="Overall Training Progress", ncols=100) as pbar_epoch:
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i in tqdm(range(num_batches), desc="Training", ncols=100):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(X_train))
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            scheduler.step(loss)
            average_epoch_loss = epoch_loss / num_batches
        pbar_epoch.update(1)

# 模型评估
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train).squeeze().cpu().numpy()
    y_pred_test = model(X_test).squeeze().cpu().numpy()
    y_pred_train = [1 if x >= 0.5 else 0 for x in y_pred_train]
    y_pred_test = [1 if x >= 0.5 else 0 for x in y_pred_test]
    # 计算模型参数
    accuracy = accuracy_score(y_test.cpu(), y_pred_test)
    precision = precision_score(y_test.cpu(), y_pred_test)
    recall = recall_score(y_test.cpu(), y_pred_test)
    f1 = f1_score(y_test.cpu(), y_pred_test)

print("简单神经网络")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# 保存模型
torch.save(model.state_dict(), 'model/cnn_model.pth')
