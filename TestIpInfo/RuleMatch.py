import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 基于规则的异常检测
def is_abnormal_packet(packet, port_usage_counts, uncommon_ports, uncommon_port_threshold=1024):
    total_data_size = packet['字节数'] / packet['包数']
    is_abnormal = False
    abnormal_type = None

    # 规则1: 数据量过大
    if total_data_size > 1200:  # 超过1200字节
        is_abnormal = True
        abnormal_type = "data_size"

    # 规则2: 工作时间规则 (比如，假设工作时间为7 AM - 22 PM)
    timestamp = packet['开始时间']
    if timestamp.hour < 7 or timestamp.hour > 21:
        if is_abnormal:
            abnormal_type += "_time"
        else:
            is_abnormal = True
            abnormal_type = "time"

    # 规则3: 端口异常检测（仅检查是否为不常见端口）
    source_port = packet['源端口']
    destination_port = packet['目的端口']

    known_ports = [20, 21, 22, 23, 25, 53, 80, 110, 139, 143, 443, 445, 3389]  # 一些熟知端口示例
    common_software_ports = [8080, 6379, 5173, 1433, 1521, 27017, 5432, 9090]  # 常用软件端口示例
    all_excluded_ports = known_ports + common_software_ports

    # 检查是否为不常见端口且不在排除列表中
    if (source_port > uncommon_port_threshold or destination_port > uncommon_port_threshold) and \
            source_port not in all_excluded_ports and destination_port not in all_excluded_ports:
        if is_abnormal:
            abnormal_type += "_port"
        else:
            is_abnormal = True
            abnormal_type = "port"

    # 规则4: TCP 标志位异常检测
    tcp_flags = packet['TCP标志位']
    # UAPRSF
    normal_flag_combinations = ['-A--S-', '-AP---', '-A----', '------', '----S-']
    if tcp_flags not in normal_flag_combinations:
        if is_abnormal:
            abnormal_type += "_flag"
        else:
            is_abnormal = True
            abnormal_type = "flag"

    return is_abnormal, abnormal_type


data_chunks = []
result_list = []
abnormal_counts = {"data_size": 0, "time": 0, "port": 0, "flag": 0, "combined": 0}

# 预计算端口相关数据结构，用于在判断每个数据包时提高效率
port_usage_counts = pd.read_csv('data/sample-data-202404/sample-data-202404.csv').groupby('源端口').size()
uncommon_ports = port_usage_counts[port_usage_counts.index > 1024].index

for chunk in pd.read_csv('data/sample-data-202404/sample-data-202404.csv', chunksize=1000, parse_dates=['开始时间']):
    print("已经处理1000条数据")
    data_chunks.append(chunk)

    for _, packet in chunk.iterrows():
        is_abnormal, abnormal_type = is_abnormal_packet(packet, port_usage_counts, uncommon_ports)
        result_list.append({'result': 1 if is_abnormal else 0})
        if is_abnormal:
            if "_" in abnormal_type:
                abnormal_counts["combined"] += 1
                for key in abnormal_counts.keys():
                    if key in abnormal_type:
                        abnormal_counts[key] += 1
            else:
                abnormal_counts[abnormal_type] += 1

print("不同类型异常的个数：")
for key, value in abnormal_counts.items():
    print(f"{key}: {value}")

result_df = pd.DataFrame(result_list)
label_df = pd.read_csv('data\\sample-data-202404\\sample-label-202404.csv')

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

print("规则匹配")
print("准确率:", accuracy)
print("召回率:", recall)
print("精确率:", precision)
print("F1值:", f1)