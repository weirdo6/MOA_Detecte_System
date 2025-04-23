import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight

# 设置 Matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    data = pd.read_csv(file_path)
    features = ['频谱质心', '频谱带宽']
    X = data[features]
    y = data['异常标签']
    return X, y

def preprocess_data(X, y):
    """
    标准化并划分为训练集和测试集，返回训练/测试数据及 scaler。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.35, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler

def oversample_data(X_train, y_train, method="SMOTE"):
    """过采样：SMOTE 或 Random Oversampling"""
    if method == "SMOTE":
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
    elif method == "Random":
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X_train, y_train)
    else:
        X_res, y_res = X_train, y_train
    print(f"过采样后训练集样本数量：{np.bincount(y_res)}")
    return X_res, y_res

def undersample_data(X_train, y_train):
    """欠采样：RandomUnderSampler"""
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    print(f"欠采样后训练集样本数量：{np.bincount(y_res)}")
    return X_res, y_res

def train_with_best_params(X_train, y_train, class_weight=None):
    """
    使用预先确定的最佳超参数训练逻辑回归模型，
    并可使用 class_weight 解决类别不均衡。
    """
    model = LogisticRegression(
        C=1,
        solver='saga',
        penalty='l1',
        max_iter=100,
        class_weight=class_weight,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"测试集召回率: {recall_score(y_test, y_pred):.4f}")
    print(f"测试集F1-score: {f1_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.show()
    return y_pred

def save_scaler(scaler, filename):
    """保存 StandardScaler"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(scaler, filename)
    print(f"Scaler 已保存为 {filename}")

def save_model(model, filename):
    """保存模型"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"最佳模型已保存为 {filename}")

def main():
    file_path = r'MOA_Detect_System_root\model_train\feature_extraction_results.csv'
    X, y = load_data(file_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    # 选择采样与类别权重策略
    sample_method    = "None"   # 可选 "SMOTE", "Random", 或 "None"
    use_class_weight= False
    do_undersample   = False

    # 过/欠采样或不采样
    if sample_method in ["SMOTE", "Random"]:
        X_train_res, y_train_res = oversample_data(X_train, y_train, method=sample_method)
    elif do_undersample:
        X_train_res, y_train_res = undersample_data(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    # 计算类别权重（若启用）
    if use_class_weight:
        weights = compute_class_weight('balanced',
                                       classes=np.unique(y_train_res),
                                       y=y_train_res)
        class_weight_dict = dict(zip(np.unique(y_train_res), weights))
        print(f"类别权重: {class_weight_dict}")
    else:
        class_weight_dict = None

    # 训练并保存模型与 scaler
    best_model = train_with_best_params(X_train_res, y_train_res, class_weight=class_weight_dict)
    save_scaler(scaler, r'MOA_Detect_System_root\models\anomaly_scaler.pkl')
    save_model(best_model, r'MOA_Detect_System_root\models\anomaly_model.pkl')

    # 交叉验证评估
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    recall_scores = cross_val_score(best_model, X_train_res, y_train_res,
                                    cv=cv, scoring='recall')
    f1_scores     = cross_val_score(best_model, X_train_res, y_train_res,
                                    cv=cv, scoring='f1')
    print("\n最终统计结果:")
    print("模型：逻辑回归（采样+最佳超参）")
    print(f"交叉验证平均召回率: {recall_scores.mean() * 100:.1f}% ± {recall_scores.std() * 100:.1f}%")
    print(f"交叉验证平均F1-score: {f1_scores.mean() * 100:.1f}%")

    # 测试集评估
    evaluate_model(best_model, X_test, y_test)

if __name__ == '__main__':
    main()
