# 数据预处理
import pandas as pd
import os
import numpy as np

from sklearn.preprocessing import LabelEncoder

# 加载数据
def load(path: str) -> pd.DataFrame:
    """加载成员A提供的预处理数据"""
    df = pd.read_csv(path, parse_dates=['start_time', 'end_time'])
    # 必须包含的字段：
    required_cols = ['src_ip',          # 源IP
                     'des_ip',          # 目的IP
                     'src_port',        # 源端口
                     'dest_port',       # 目的端口
                     'tcp_flag',        # TCP标志位
                     'protocol',        # 协议类型
                     'tls_version',     # TLS版本
                     'dns_query_type',  # DNS查询类型
                     'http_method',     # HTTP方法
                     'icmp_type_code',  # ICMP类型代码组合
                     'as_path',         # AS路径
                     'ttl_hop',         # TTL跳数
                     'start_time',      # 开始时间
                     'end_time',        # 结束时间
                     'packs_count',     # 包数
                     'bytes_count']     # 字节数
    assert set(required_cols).issubset(df.columns)
    return df

# 数据预处理
def preprocess(df):
    # 转换时间列为时间戳
    df['start_time'] = pd.to_datetime(df['start_time']).astype(np.int64) // 10 ** 9
    df['end_time'] = pd.to_datetime(df['end_time']).astype(np.int64) // 10 ** 9

    # 计算会话持续时间
    df['duration'] = df['end_time'] - df['start_time']

    # 对分类特征进行编码
    categorical_cols = ['tcp_flag', 'protocol', 'tls_version', 'dns_query_type',
                        'http_method', 'icmp_type_code']

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # IP地址转换为数值
    def ip_to_int(ip):
        return sum([int(x) * (256 ** i) for i, x in enumerate(reversed(ip.split('.')))])

    df['src_ip'] = df['src_ip'].apply(ip_to_int)
    df['des_ip'] = df['des_ip'].apply(ip_to_int)

    # AS路径处理
    df['as_path'] = df['as_path'].apply(lambda x: len(str(x).split()))

    return df
