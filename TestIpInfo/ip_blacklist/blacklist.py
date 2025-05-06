import mysql.connector
import pandas as pd

# 定义黑名单规则常量
BLACKLIST_RULES = {
    'min_count': 3,        # 最小异常次数
    'time_window': 24,     # 时间窗口(小时)
    'min_confidence': 0.8, # 最小置信度
    'min_severity': 0.7    # 最小严重程度
}

class BlacklistManager:
    def __init__(self, host="localhost", user="root", password="123456", database="network_security"):
        """初始化黑名单管理器并建立数据库连接"""
        try:
            self.conn = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            self.cursor = self.conn.cursor()
            print("数据库连接成功")

            # self._init_tables()  # 初始化数据表

        except mysql.connector.Error as err:
            print(f"数据库连接失败: {err}")
            raise



    def query_blacklist(self):
        """查询黑名单IP列表"""
        try:
            query = "SELECT ip FROM blacklist"
            df = pd.read_sql_query(query, self.conn)
            return df['ip'].tolist()

        except Exception as e:
            print(f"查询黑名单失败: {e}")
            return []

    def batch_insert_records(self, records):
        """批量插入或更新异常记录，使用加权平均更新confidence和severity
        Args:
            records (list): [(ip, anomaly_count, time_window, confidence, severity), ...]
        """
        try:
            # 转换记录中的numpy类型为Python原生类型
            converted_records = [
                (
                    str(ip),  # 确保IP是字符串
                    int(count),  # 转换为Python int
                    int(window),  # 转换为Python int
                    float(conf),  # 转换为Python float
                    float(sev)  # 转换为Python float
                )
                for ip, count, window, conf, sev in records
            ]

            # 准备更新语句
            insert_query = '''
                INSERT INTO ip_anomaly_records
                (ip, anomaly_count, time_window, confidence, severity)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                anomaly_count = anomaly_count + VALUES(anomaly_count),
                time_window = VALUES(time_window),
                confidence = (anomaly_count * confidence + VALUES(anomaly_count) * VALUES(confidence)) / (anomaly_count + VALUES(anomaly_count)),
                severity = (anomaly_count * severity + VALUES(anomaly_count) * VALUES(severity)) / (anomaly_count + VALUES(anomaly_count))
            '''

            # 执行批量插入/更新
            self.cursor.executemany(insert_query, converted_records)
            self.conn.commit()
            return True

        except Exception as e:
            print(f"批量插入记录失败: {e}")
            self.conn.rollback()
            return False

    def update_blacklist(self):
        """根据固定规则更新黑名单"""
        try:
            query = '''
                SELECT ip
                FROM ip_anomaly_records
                WHERE anomaly_count >= %s
                    AND time_window <= %s
                    AND confidence >= %s
                    AND severity >= %s
                GROUP BY ip
            '''

            params = (BLACKLIST_RULES['min_count'],
                     BLACKLIST_RULES['time_window'],
                     BLACKLIST_RULES['min_confidence'],
                     BLACKLIST_RULES['min_severity'])

            self.cursor.execute(query, params)
            results = self.cursor.fetchall()

            if results:
                # 批量添加到黑名单（忽略已存在的IP）
                insert_query = '''
                    INSERT IGNORE INTO blacklist (ip)
                    VALUES (%s)
                '''
                self.cursor.executemany(insert_query, results)
                self.conn.commit()
                return len(results)

            return 0

        except Exception as e:
            print(f"更新黑名单失败: {e}")
            self.conn.rollback()
            return 0
    def filter_blacklist(self, df):
        """根据黑名单过滤数据"""
        try:
            blacklist_ips = self.query_blacklist()
            filtered_df = df[~df['src_ip'].isin(blacklist_ips) & ~df['des_ip'].isin(blacklist_ips)]
            return filtered_df
        except Exception as e:
            print(f"过滤黑名单失败: {e}")
            return df

    def update_blacklist_by_df(self, df):
        # 1. 读取黑名单过滤
        blacklist_ips = self.query_blacklist()

        # 找出与黑名单IP通信的非黑名单IP
        suspicious_ips = []

        # 检查源IP在黑名单中的情况
        mask_src_blacklist = df['src_ip'].isin(blacklist_ips)
        suspicious_ips.extend(
            df[mask_src_blacklist]['des_ip'].unique().tolist()
        )

        # 检查目的IP在黑名单中的情况
        mask_dest_blacklist = df['des_ip'].isin(blacklist_ips)
        suspicious_ips.extend(
            df[mask_dest_blacklist]['src_ip'].unique().tolist()
        )

        # 移除已在黑名单中的IP
        suspicious_ips = list(set(suspicious_ips) - set(blacklist_ips))

        # 准备要插入异常记录表的数据,直接置为阈值，方便后面加入黑名单表
        suspicious_records = [
            (ip, BLACKLIST_RULES['min_count'], BLACKLIST_RULES['time_window'], BLACKLIST_RULES['min_confidence'], BLACKLIST_RULES['min_severity'])  # 示例置信度和严重度
            for ip in suspicious_ips
        ]

        # 插入异常记录
        if suspicious_records:
            self.batch_insert_records(suspicious_records)
            print(f"发现{len(suspicious_records)}个可疑IP与黑名单IP有通信")

    def update_blacklist_by_rule_anomaly(self, df):
        """根据规则异常数据更新黑名单"""
        try:
            # 分别处理源IP和目的IP
            records = []
            for _, row in df.iterrows():
                # 添加源IP记录
                records.append(
                    (row['src_ip'], 1, BLACKLIST_RULES['time_window'], 0.6, 0.7)
                )
                # 添加目的IP记录
                records.append(
                    (row['des_ip'], 1, BLACKLIST_RULES['time_window'], 0.5, 0.5)
                )

            if records:
                self.batch_insert_records(records)
                print(f"根据规则异常数据更新黑名单成功，处理了{len(records)}条IP记录")
        except Exception as e:
            print(f"根据规则异常数据更新黑名单失败: {e}")

    def update_blacklist_by_model_anomaly(self, df):
        """根据模型异常数据更新黑名单"""
        try:
            # 分别处理源IP和目的IP
            records = []
            for _, row in df.iterrows():
                # 添加源IP记录
                records.append(
                    (row['src_ip'], 1, BLACKLIST_RULES['time_window'], 0.9, 0.9)
                )
                # 添加目的IP记录
                records.append(
                    (row['des_ip'], 1, BLACKLIST_RULES['time_window'], 0.8, 0.9)
                )

            if records:
                self.batch_insert_records(records)
                print(f"根据模型异常数据更新黑名单成功，处理了{len(records)}条IP记录")
        except Exception as e:
            print(f"根据模型异常数据更新黑名单失败: {e}")




    def __del__(self):
        """析构函数：关闭数据库连接"""
        try:
            if hasattr(self, 'cursor') and self.cursor:
                self.cursor.close()
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
                print("数据库连接已关闭")
        except Exception as e:
            print(f"关闭数据库连接失败: {e}")