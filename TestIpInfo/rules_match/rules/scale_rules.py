"""规模规则配置文件"""

# 默认规模规则
DEFAULT_SCALE_RULES = {
    "packs_count": {
        "min": 1,
        "max": 10000
    },
    "bytes_count": {
        "min": 64,
        "max": 1000000
    },
    "burst_threshold": {
        "packs_per_second": 100,
        "bytes_per_second": 1000000
    }
}

# 低流量规则
LOW_TRAFFIC_RULES = {
    "packs_count": {
        "min": 1,
        "max": 1000
    },
    "bytes_count": {
        "min": 64,
        "max": 100000
    },
    "burst_threshold": {
        "packs_per_second": 10,
        "bytes_per_second": 100000
    }
}

# 高流量规则
HIGH_TRAFFIC_RULES = {
    "packs_count": {
        "min": 1,
        "max": 100000
    },
    "bytes_count": {
        "min": 64,
        "max": 10000000
    },
    "burst_threshold": {
        "packs_per_second": 1000,
        "bytes_per_second": 10000000
    }
} 