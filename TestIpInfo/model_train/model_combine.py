import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib


def ensemble_predict(X_scaled, rf_model, xgb_model):
    """1. 模型融合: 投票机制
    基于评估结果，两个模型性能相近，使用简单多数投票"""

    # 获取两个模型的预测结果
    rf_pred = rf_model.predict(X_scaled)
    xgb_pred = xgb_model.predict(X_scaled)

    # 多数投票 (至少一个预测为异常则判定为异常)
    final_pred = (rf_pred + xgb_pred) >= 1

    return final_pred


def stacking_predict(X_scaled, rf_model, xgb_model):
    """2. 堆叠: 使用逻辑回归作为二级模型"""
    try:
        # 获取基模型预测概率
        rf_proba = rf_model.predict_proba(X_scaled)[:, 1].reshape(-1, 1)
        xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1].reshape(-1, 1)

        # 组合预测概率作为新特征
        meta_features = np.hstack([rf_proba, xgb_proba])

        try:
            # 尝试加载元模型
            meta_model = joblib.load("../model/meta_model.joblib")
            print("成功加载元模型")
            final_pred = meta_model.predict(meta_features)
        except (FileNotFoundError, OSError) as e:
            print(f"加载元模型失败: {e}")
            print("使用平均概率作为备选方案")
            # 使用平均概率作为备选
            final_pred = (meta_features.mean(axis=1) >= 0.5)

        return final_pred.astype(bool)

    except Exception as e:
        print(f"堆叠预测出错: {e}")
        # 返回保守的预测结果
        return np.zeros(len(X_scaled), dtype=bool)


def weighted_average_predict(X_scaled, rf_model, xgb_model):
    """3. 加权平均: 基于模型评估指标确定权重"""

    # 基于F1分数确定权重 (RF: 0.92, XGB: 0.92)
    rf_weight = 0.5  # RF权重
    xgb_weight = 0.5  # XGB权重

    # 获取预测概率
    rf_proba = rf_model.predict_proba(X_scaled)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]

    # 加权平均
    weighted_proba = (rf_weight * rf_proba + xgb_weight * xgb_proba)

    # 使用0.5作为阈值
    final_pred = weighted_proba >= 0.5

    return final_pred


def train_meta_model(X_train_scaled, y_train, rf_model, xgb_model):
    """训练堆叠模型的元模型"""
    try:
        # 获取基模型预测概率
        rf_proba = rf_model.predict_proba(X_train_scaled)[:, 1].reshape(-1, 1)
        xgb_proba = xgb_model.predict_proba(X_train_scaled)[:, 1].reshape(-1, 1)

        # 组合预测概率
        meta_features = np.hstack([rf_proba, xgb_proba])

        # 训练元模型
        meta_model = LogisticRegression(random_state=42)
        meta_model.fit(meta_features, y_train)

        # 保存元模型
        try:
            model_dir = "../model"
            joblib.dump(meta_model, f"{model_dir}/meta_model.joblib")
            print("元模型保存成功")
        except Exception as e:
            print(f"保存元模型失败: {e}")

        return meta_model

    except Exception as e:
        print(f"训练元模型失败: {e}")
        return None