import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from datetime import datetime
from common.load_data import preprocess, load
from tqdm import tqdm

from model_train.model_combine import train_meta_model


# Random Forest + XGBoost 模型训练
def train_models():
    # 读取数据 (目前小数据集 只有一万条)
    print("正在加载数据...")
    df = load("../data/fake_data/model_train_data.csv")

    # 数据预处理
    print("正在预处理数据...")
    df = preprocess(df)

    # 分离特征和标签
    X = df.drop(['label'], axis=1)
    y = df['label']

    # 划分训练集和测试集 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 确保model目录存在
    model_dir = "../model"
    os.makedirs(model_dir, exist_ok=True)

    # 训练Random Forest
    print("正在训练Random Forest模型...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)

    # 训练XGBoost
    print("正在训练XGBoost模型...")
    xgb_model = XGBClassifier(n_estimators=100, random_state=42,
                              scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
    xgb_model.fit(X_train_scaled, y_train)

    # 训练堆叠模型的元模型
    print("正在训练堆叠模型的元模型...")
    meta_model = train_meta_model(X_train_scaled, y_train, rf_model, xgb_model)
    meta_model.fit(X_train_scaled, y_train)


    # 模型评估
    models = {'Random Forest': rf_model, 'XGBoost': xgb_model, 'Stacking': meta_model}

    for name, model in models.items():
        print(f"\n{name} 模型评估结果:")
        y_pred = model.predict(X_test_scaled)
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        print("\n混淆矩阵:")
        print(confusion_matrix(y_test, y_pred))

    # 保存模型和scaler
    for file_name, obj in tqdm([
        (f'rf_model.joblib', rf_model),
        (f'xgb_model.joblib', xgb_model),
        (f'scaler.joblib', scaler)
    ], desc="Saving models"):
        save_path = os.path.join(model_dir, file_name)
        joblib.dump(obj, save_path)
        print(f"已保存：{save_path}")








if __name__ == "__main__":
    train_models()

