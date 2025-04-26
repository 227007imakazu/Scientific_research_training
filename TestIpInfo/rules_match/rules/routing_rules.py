"""路由规则配置文件"""

# 默认路由规则
DEFAULT_ROUTING_RULES = {
    "ttl_hop": {
        "min": 1,
        "max": 64
    },
    "as_path": {
        "min": 1,
        "max": 10
    },
    "src_ip_blocked": set(),  # 源IP黑名单
    "des_ip_blocked": set(),  # 目的IP黑名单
    "src_ip_allowed": set(),  # 源IP白名单
    "des_ip_allowed": set(),  # 目的IP白名单
    "src_port_suspicious": {80, 443, 53, 22, 3389},  # 可疑源端口
    "dest_port_suspicious": {80, 443, 53, 22, 3389}  # 可疑目的端口
}

# 内部网络规则
INTERNAL_NETWORK_RULES = {
    "ttl_hop": {
        "min": 1,
        "max": 30
    },
    "as_path": {
        "min": 1,
        "max": 5
    },
    "src_ip_blocked": {"192.168.1.100", "10.0.0.1"},
    "des_ip_blocked": {"192.168.1.100", "10.0.0.1"},
    "src_ip_allowed": {"192.168.1.0/24", "10.0.0.0/8"},
    "des_ip_allowed": {"192.168.1.0/24", "10.0.0.0/8"},
    "src_port_suspicious": {22, 3389, 445, 135, 139},
    "dest_port_suspicious": {22, 3389, 445, 135, 139}
}

# DMZ规则
DMZ_RULES = {
    "ttl_hop": {
        "min": 5,
        "max": 64
    },
    "as_path": {
        "min": 1,
        "max": 15
    },
    "src_ip_blocked": set(),
    "des_ip_blocked": set(),
    "src_ip_allowed": {"203.0.113.0/24"},
    "des_ip_allowed": {"203.0.113.0/24"},
    "src_port_suspicious": {80, 443, 25, 465, 587},
    "dest_port_suspicious": {80, 443, 25, 465, 587}
} 