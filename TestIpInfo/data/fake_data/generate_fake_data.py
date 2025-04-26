import numpy as np
import pandas as pd

import random
from datetime import datetime, timedelta
import os


def generate_fake_data(num_samples: int, anomaly_ratio: float = 0.2) -> pd.DataFrame:
    """
    生成带有标签的假网络数据
    :param num_samples: 样本数量
    :param anomaly_ratio: 异常数据比例，默认为20%
    :return: 包含标签的DataFrame
    """
    # 定义必需字段列表
    required_cols = ['src_ip', 'des_ip', 'src_port', 'dest_port', 'tcp_flag', 'protocol',
                    'tls_version', 'dns_query_type', 'http_method', 'icmp_type_code',
                    'as_path', 'ttl_hop', 'start_time', 'end_time', 'packs_count', 
                    'bytes_count', 'label']

    # 生成随机数据
    data = {
        'src_ip': [
            f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}" 
            for _ in range(num_samples)],
        'des_ip': [
            f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}" 
            for _ in range(num_samples)],
        'src_port': np.random.randint(1, 65535, num_samples),
        'dest_port': np.random.randint(1, 65535, num_samples),
        'tcp_flag': np.random.choice(['SYN', 'ACK', 'FIN', 'RST', 'PSH', 'URG'], num_samples),
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], num_samples),
        'tls_version': np.random.choice(['TLS 1.0', 'TLS 1.1', 'TLS 1.2', 'TLS 1.3'], num_samples),
        'dns_query_type': np.random.choice(['A', 'AAAA', 'CNAME', 'MX', 'TXT'], num_samples),
        'http_method': np.random.choice(['GET', 'POST', 'PUT', 'DELETE', 'HEAD'], num_samples),
        'icmp_type_code': np.random.choice(['Echo Request', 'Echo Reply', 'Destination Unreachable'], num_samples),
        'as_path': [f"{random.randint(100, 65000)} {random.randint(100, 65000)}" for _ in range(num_samples)],
        'ttl_hop': np.random.randint(1, 255, num_samples),
        'start_time': [datetime.now() - timedelta(seconds=random.randint(1, 3600)) for _ in range(num_samples)],
        'end_time': [datetime.now() - timedelta(seconds=random.randint(1, 3600)) for _ in range(num_samples)],
        'packs_count': np.random.randint(1, 1000, num_samples),
        'bytes_count': np.random.randint(100, 10000, num_samples),
        'label': [0] * num_samples  # 初始全部设为0(正常)
    }

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 确保start_time早于end_time
    for i in range(num_samples):
        if df.at[i, 'start_time'] > df.at[i, 'end_time']:
            df.at[i, 'end_time'] = df.at[i, 'start_time'] + timedelta(seconds=random.randint(1, 3600))

    # 随机选择一部分数据标记为异常(1)
    num_anomalies = int(num_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
    
    # 修改异常数据的特征
    for idx in anomaly_indices:
        df.at[idx, 'label'] = 1  # 异常标记为1
        # 异常数据通常有异常特征，随机修改一些特征
        if random.random() > 0.5:
            df.at[idx, 'src_port'] = random.choice([22, 80, 443, 3389, 8080])  # 常见端口
        if random.random() > 0.5:
            df.at[idx, 'dest_port'] = random.choice([22, 80, 443, 3389, 8080])
        if random.random() > 0.5:
            df.at[idx, 'tcp_flag'] = random.choice(['SYN', 'RST', 'FIN'])  # 攻击常见标志
        if random.random() > 0.5:
            df.at[idx, 'packs_count'] = random.randint(1000, 10000)  # 异常高的包数量
        if random.random() > 0.5:
            df.at[idx, 'bytes_count'] = random.randint(10000, 100000)  # 异常高的字节数

    return df


# 生成100条假数据，其中20%为异常
fake_data = generate_fake_data(10000)

# 保存到同级目录的CSV文件
csv_filename = "model_train_data.csv"

# 提取目录路径
directory = os.path.dirname(csv_filename)

# 保存数据到 CSV 文件
fake_data.to_csv(csv_filename, index=False)
print(f"假数据已保存到 {csv_filename}")