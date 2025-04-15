from flask import Flask, jsonify
# 创建Flask应用
app = Flask(__name__)
def RuleMatch_model():
    accuracy = 0.434
    precision = 0.308
    recall = 0.672
    f1 = 0.422
    return accuracy, precision, recall, f1
def IsolationForest_model():
    accuracy = 0.864
    precision = 0.784
    recall = 0.788
    f1 = 0.786
    return accuracy, precision, recall, f1
def CNN_model():
    # 200 10000
    accuracy = 0.686
    precision = 0.303
    recall = 0.016
    f1 = 0.031
    return accuracy, precision, recall, f1
def RuleMatchBie():
    data_size = 83686
    time = 465103
    port = 287765
    flag = 16730
    return data_size, time, port, flag
# 定义路由

# 返回规则匹配的四个参数
@app.route('/ruleMatch_v')
def get_ruleMatch_model_metrics():
    accuracy, precision, recall, f1 = RuleMatch_model()
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    return jsonify(metrics)
# 返回孤立森林的四个参数
@app.route('/isolationForest_v')
def get_isolationForest_model_metrics():
    accuracy, precision, recall, f1 = IsolationForest_model()
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    return jsonify(metrics)
# 返回CNN的四个参数
@app.route('/cnn_v')
def get_cnn_model_metrics():
    accuracy, precision, recall, f1 = CNN_model()
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    return jsonify(metrics)
# 返回规则匹配的不同异常类型数量
@app.route('/ruleMatch_c')
def get_ruleMatch_model_count():
    data_size, time, port, flag=RuleMatchBie()
    metrics = {
        "size": data_size,
        "time": time,
        "port": port,
        "flag": flag
    }
    return jsonify(metrics)







# 启动应用
if __name__ == '__main__':
    app.run(debug=True)