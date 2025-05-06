from ip_blacklist.blacklist import BlacklistManager

# 创建实例
blacklist_mgr = BlacklistManager()

# 查询黑名单IP列表
blacklist_ips = blacklist_mgr.query_blacklist()

print("当前黑名单IP列表:", blacklist_ips)

# 批量插入异常记录
records = [
    ('192.168.1.1', 1, 24, 0.9, 0.8),
    ('192.168.1.1', 1, 24, 0.9, 0.8),
    ('192.168.1.2', 3, 24, 0.95, 0.9)
]
blacklist_mgr.batch_insert_records(records)


updated_count = blacklist_mgr.update_blacklist()