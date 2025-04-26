"""时序规则配置文件"""
from datetime import time

# 默认时序规则
DEFAULT_TIME_RULES = {
    "time_range": {
        "start_time": time(0, 0),  # 默认开始时间
        "end_time": time(23, 59),  # 默认结束时间
    },
    "duration_threshold": 3600,  # 默认持续时间阈值（秒）
}

# 工作时间规则
BUSINESS_HOURS_RULES = {
    "time_range": {
        "start_time": time(8, 0),  # 工作开始时间
        "end_time": time(18, 0),  # 工作结束时间
    },
    "duration_threshold": 7200,  # 2小时
}

# 夜间规则
NIGHT_HOURS_RULES = {
    "time_range": {
        "start_time": time(18, 0),  # 夜间开始时间
        "end_time": time(8, 0),  # 夜间结束时间
    },
    "duration_threshold": 1800,  # 30分钟
} 