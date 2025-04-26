from typing import Dict, List, Any, Tuple
import pandas as pd
from datetime import datetime, time
from .rules import (
    DEFAULT_TIME_RULES,
    DEFAULT_PROTOCOL_RULES,
    DEFAULT_ROUTING_RULES,
    DEFAULT_SCALE_RULES
)

class RuleEngine:
    def __init__(self):
        # 初始化各类规则为默认值
        self.time_rules = DEFAULT_TIME_RULES.copy()
        self.protocol_rules = DEFAULT_PROTOCOL_RULES.copy()
        self.routing_rules = DEFAULT_ROUTING_RULES.copy()
        self.scale_rules = DEFAULT_SCALE_RULES.copy()

    def check_rules(self, row: pd.Series) -> Tuple[bool, List[str]]:
        """检查所有规则并返回结果和违规原因"""
        is_normal = True
        violation_reasons = []

        # 检查时序规则
        if not self.check_time_rules(row):
            is_normal = False
            violation_reasons.append("time_rules")

        # 检查协议规则
        if not self.check_protocol_rules(row):
            is_normal = False
            violation_reasons.append("protocol_rules")

        # 检查路由规则
        if not self.check_routing_rules(row):
            is_normal = False
            violation_reasons.append("routing_rules")

        # 检查规模规则
        if not self.check_scale_rules(row):
            is_normal = False
            violation_reasons.append("scale_rules")

        return is_normal, violation_reasons

    def check_time_rules(self, row: pd.Series) -> bool:
        """检查时序规则"""
        start_time = pd.to_datetime(row['start_time']).time()
        end_time = pd.to_datetime(row['end_time']).time()
        
        # 检查时间范围
        if not (self.time_rules["time_range"]["start_time"] <= start_time <= self.time_rules["time_range"]["end_time"]):
            return False
            
        # 检查持续时间
        duration = (pd.to_datetime(row['end_time']) - pd.to_datetime(row['start_time'])).total_seconds()
        if duration > self.time_rules["duration_threshold"]:
            return False
            
        return True

    def check_protocol_rules(self, row: pd.Series) -> bool:
        """检查协议规则"""
        # 检查协议类型
        if pd.notna(row['protocol']) and row['protocol'] not in self.protocol_rules["protocol"]:
            return False
            
        # 检查TLS版本
        if pd.notna(row['tls_version']) and row['tls_version'] not in self.protocol_rules["tls_version"]:
            return False
            
        # 检查HTTP方法
        if pd.notna(row['http_method']) and row['http_method'] not in self.protocol_rules["http_method"]:
            return False
            
        # 检查DNS查询类型
        if pd.notna(row['dns_query_type']) and row['dns_query_type'] not in self.protocol_rules["dns_query_type"]:
            return False
            
        # 检查TCP标志
        if pd.notna(row['tcp_flag']) and row['tcp_flag'] not in self.protocol_rules["tcp_flag"]:
            return False
            
        return True

    def check_routing_rules(self, row: pd.Series) -> bool:
        """检查路由规则"""
        # 检查TTL范围
        if not (self.routing_rules["ttl_hop"]["min"] <= row['ttl_hop'] <= self.routing_rules["ttl_hop"]["max"]):
            return False
            
        # 检查AS路径长度
        if pd.notna(row['as_path']):
            as_path = len(str(row['as_path']).split())
            if not (self.routing_rules["as_path"]["min"] <= as_path <= self.routing_rules["as_path"]["max"]):
                return False
                
        # 检查源IP黑白名单
        if row['src_ip'] in self.routing_rules["src_ip_blocked"]:
            return False
        if self.routing_rules["src_ip_allowed"] and row['src_ip'] not in self.routing_rules["src_ip_allowed"]:
            return False
            
        # 检查目的IP黑白名单
        if row['des_ip'] in self.routing_rules["des_ip_blocked"]:
            return False
        if self.routing_rules["des_ip_allowed"] and row['des_ip'] not in self.routing_rules["des_ip_allowed"]:
            return False
            
        # 检查可疑端口
        if row['src_port'] in self.routing_rules["src_port_suspicious"]:
            return False
        if row['dest_port'] in self.routing_rules["dest_port_suspicious"]:
            return False
            
        return True

    def check_scale_rules(self, row: pd.Series) -> bool:
        """检查规模规则"""
        # 检查包数范围
        if not (self.scale_rules["packs_count"]["min"] <= row['packs_count'] <= self.scale_rules["packs_count"]["max"]):
            return False
            
        # 检查字节数范围
        if not (self.scale_rules["bytes_count"]["min"] <= row['bytes_count'] <= self.scale_rules["bytes_count"]["max"]):
            return False
            
        # 检查突发流量
        duration = (pd.to_datetime(row['end_time']) - pd.to_datetime(row['start_time'])).total_seconds()
        if duration > 0:
            packs_per_second = row['packs_count'] / duration
            bytes_per_second = row['bytes_count'] / duration
            
            if (packs_per_second > self.scale_rules["burst_threshold"]["packs_per_second"] or
                bytes_per_second > self.scale_rules["burst_threshold"]["bytes_per_second"]):
                return False
                
        return True

    def apply_rules(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        应用所有规则到数据框
        返回: (正常数据, 异常数据)
        """
        # 创建结果列
        df['is_normal'] = True
        df['violation_reasons'] = ''

        # 应用规则检查
        for idx, row in df.iterrows():
            is_normal, reasons = self.check_rules(row)
            df.at[idx, 'is_normal'] = is_normal
            df.at[idx, 'violation_reasons'] = ','.join(reasons) if reasons else ''

        # 分离正常和异常数据
        normal_data = df[df['is_normal']].drop(['is_normal', 'violation_reasons'], axis=1)
        anomaly_data = df[~df['is_normal']].drop(['is_normal'], axis=1)

        # 输出统计信息
        print(f"\n规则过滤结果:")
        print(f"正常数据: {len(normal_data)}条")
        print(f"异常数据: {len(anomaly_data)}条")

        return normal_data, anomaly_data

    def update_rules(self, rule_type: str, new_rules: Dict[str, Any]) -> None:
        """更新规则配置"""
        if rule_type == "time":
            self.time_rules.update(new_rules)
        elif rule_type == "protocol":
            self.protocol_rules.update(new_rules)
        elif rule_type == "routing":
            self.routing_rules.update(new_rules)
        elif rule_type == "scale":
            self.scale_rules.update(new_rules)
        else:
            raise ValueError(f"Unknown rule type: {rule_type}") 