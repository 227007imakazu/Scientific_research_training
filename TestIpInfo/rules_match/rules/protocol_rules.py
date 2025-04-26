"""协议规则配置文件"""

# 默认协议规则
DEFAULT_PROTOCOL_RULES = {
    "protocol": ["TCP", "UDP", "ICMP"],
    "tls_version": ["TLS 1.2", "TLS 1.3"],
    "http_method": ["GET", "POST", "PUT", "DELETE"],
    "dns_query_type": ["A", "AAAA", "MX", "TXT"],
    "tcp_flag": ["ACK", "SYN", "FIN", "RST"]
}

# 严格安全规则
STRICT_SECURITY_RULES = {
    "protocol": ["TCP"],
    "tls_version": ["TLS 1.3"],
    "http_method": ["GET", "POST"],
    "dns_query_type": ["A", "AAAA"],
    "tcp_flag": ["ACK", "SYN", "FIN"]
}

# Web应用规则
WEB_APP_RULES = {
    "protocol": ["TCP"],
    "tls_version": ["TLS 1.2", "TLS 1.3"],
    "http_method": ["GET", "POST", "PUT", "DELETE", "PATCH"],
    "dns_query_type": ["A", "AAAA", "CNAME"],
    "tcp_flag": ["ACK", "SYN", "FIN", "RST", "PSH"]
} 