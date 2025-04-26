import joblib

from common.load_data import preprocess
from model_train.model_combine import ensemble_predict, stacking_predict, weighted_average_predict


def detect_anomalies(filtered_df,method='ensemble'):
    """
    对规则过滤后的数据进行异常检测
    method:'ensemble' or 'stacking' or 'weighted'
    """
    try:
        # 加载模型和scaler
        model_dir = "../model"
        rf_model = joblib.load(f"{model_dir}/rf_model.joblib")
        xgb_model = joblib.load(f"{model_dir}/xgb_model.joblib")
        scaler = joblib.load(f"{model_dir}/scaler.joblib")

        # 数据预处理
        X = preprocess(filtered_df)

        # 删除不需要的列（如果有）
        if 'label' in X.columns:
            X = X.drop(['label'], axis=1)

        # 特征标准化
        X_scaled = scaler.transform(X)

        # 根据选择的方法进行预测
        if method == 'ensemble':
            final_pred = ensemble_predict(X_scaled, rf_model, xgb_model)
        elif method == 'stacking':
            final_pred = stacking_predict(X_scaled, rf_model, xgb_model)
        elif method == 'weighted':
            final_pred = weighted_average_predict(X_scaled, rf_model, xgb_model)
        else:
            raise ValueError("不支持的组合方法")


        # 分离正常和异常数据
        filtered_df['is_anomaly'] = final_pred
        normal_data = filtered_df[~filtered_df['is_anomaly']].drop('is_anomaly', axis=1)
        anomaly_data = filtered_df[filtered_df['is_anomaly']].drop('is_anomaly', axis=1)

        # 输出检测结果统计
        print(f"\n模型检测结果:")
        print(f"正常数据: {len(normal_data)}条")
        print(f"异常数据: {len(anomaly_data)}条")

        return normal_data, anomaly_data

    except Exception as e:
        print(f"异常检测出错: {str(e)}")
        return None, None


