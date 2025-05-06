from ip_blacklist.blacklist import BlacklistManager
from model_train.detect import detect_anomalies
from rules_match.rule_engine import RuleEngine
import pandas as pd
from datetime import datetime, timedelta
from rules_match.rules import (
    BUSINESS_HOURS_RULES,
    STRICT_SECURITY_RULES,
    INTERNAL_NETWORK_RULES,
    LOW_TRAFFIC_RULES,
    DEFAULT_TIME_RULES,
    DEFAULT_PROTOCOL_RULES,
    DEFAULT_ROUTING_RULES,
    DEFAULT_SCALE_RULES
)
from common.load_data import load

def main():
    # 创建规则引擎实例
    engine = RuleEngine()

    # 创建数据库实例
    blacklist_mgr = BlacklistManager()

    
    # 使用预定义的规则集
    engine.update_rules("time", DEFAULT_TIME_RULES)
    engine.update_rules("protocol", DEFAULT_PROTOCOL_RULES)
    engine.update_rules("routing", DEFAULT_ROUTING_RULES)
    engine.update_rules("scale", DEFAULT_SCALE_RULES)

    
    # 读取CSV数据
    try:
        df=load("../data/fake_data/model_train_data.csv")

        # 1. 读取黑名单过滤
        blacklist_mgr.update_blacklist_by_df(df) # 更新黑名单
        df = blacklist_mgr.filter_blacklist(df)  # 只保留src和des都不在黑名单的数据

        # 2. 应用规则过滤
        normal_data, rule_anomaly_data = engine.apply_rules(df)

        # 3. 模型过滤
        # method: 'ensemble' or 'stacking' or 'weighted'
        model_normal_data, model_anomaly_data = detect_anomalies(normal_data, method='stacking')

        # 4. 异常数据-结合系数判断是否入黑名单
        blacklist_mgr.update_blacklist_by_rule_anomaly(model_anomaly_data)  # 规则异常
        blacklist_mgr.update_blacklist_by_model_anomaly(model_anomaly_data)  # 模型检测异常

        # 5. 输出过滤结果/异常结果
        print(f"\n过滤统计:")
        print(f"原始数据: {len(df)}条")
        print(f"规则过滤异常: {len(rule_anomaly_data)}条")
        print(f"模型检测异常: {len(model_anomaly_data)}条")
        print(f"最终正常数据: {len(model_normal_data)}条")







        
        # 保存过滤后的结果
        # filtered_df.to_csv("../data/filtered_network_data.csv", index=False)
        # print("过滤后的数据已保存到 filtered_network_data.csv")
        
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")

if __name__ == "__main__":
    main() 