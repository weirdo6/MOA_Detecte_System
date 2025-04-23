import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    recall_score, f1_score
)
import joblib
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
# 设置 Matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
class SVMTrainer:
    def __init__(self, seed: int = 42):
        self.set_random_seed(seed)
        self.scaler = StandardScaler()
        self.best_model = None

    @staticmethod
    def set_random_seed(seed: int):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

    def load_and_preprocess_data(self, path, features, label_col='缺陷类型', test_ratio=0.35):
        data = pd.read_csv(path)
        data = data[data[label_col] != 'Normal_Data_Peaks']  # 去除不需要的标签
        X = data[features].values

        y = data[label_col].astype('category').cat.codes.values  # 转换为分类标签

        # 拆分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, stratify=y, random_state=42
        )

        # 数据标准化
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, y_train, X_test, y_test

    def oversample_data(self, X_train, y_train, method="SMOTE"):
        """
        过采样方法：SMOTE、Random或无采样
        """
        if method == "SMOTE":
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        elif method == "Random":
            oversampler = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train
        print(f"过采样后训练集样本数量：{np.bincount(y_resampled)}")
        return X_resampled, y_resampled

    def undersample_data(self, X_train, y_train):
        """
        欠采样方法
        """
        undersampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
        print(f"欠采样后训练集样本数量：{np.bincount(y_resampled)}")
        return X_resampled, y_resampled

    def train_with_best_params(self, X_train, y_train, class_weight=None, model_dir='trained_models'):
        os.makedirs(model_dir, exist_ok=True)

        # 使用已确定的最佳参数训练SVM
        svm = SVC(C=1, gamma=1, kernel='rbf', probability=True, random_state=42, class_weight=class_weight)
        svm.fit(X_train, y_train)
        
        # 保存模型及scaler
        joblib.dump(svm, os.path.join(model_dir, 'defect_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'defect_scaler.pkl'))
        self.best_model = svm

        print("\n=== 模型最终统计结果 ===")
        print("模型：SVM")
        print(f"最佳参数组合：{{'C': 1, 'gamma': 1, 'kernel': 'rbf'}}")

        return svm

    def evaluate_on_test_set(self, X_test, y_test):
        if self.best_model is None:
            raise ValueError("请先训练模型")

        y_pred = self.best_model.predict(X_test)

        # 使用加权平均来处理多分类任务
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='weighted')  # 改为加权平均
        f1 = f1_score(y_test, y_pred, average='weighted')  # 改为加权平均
        report = classification_report(y_test, y_pred, digits=4)
        cm = confusion_matrix(y_test, y_pred)

        print(f"测试集准确率: {acc:.4f}")
        print(f"测试集召回率: {recall:.4f}")
        print(f"测试集F1-score: {f1:.4f}\n")

        print("分类报告：")
        print(report)

        self.plot_confusion_matrix(cm, title="SVM 混淆矩阵")

        return acc, recall, f1, report, cm

    @staticmethod
    def plot_confusion_matrix(cm, title="Confusion Matrix"):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.show()

def main():
    config = {
        'data_path': r'MOA_Detect_System_root\model_train\feature_extraction_results.csv',
        'features': ['频谱熵', '峰峰值', '最大值', '小波能量', '最小值', '标准差'],
        'test_ratio': 0.35,
        'model_dir': 'MOA_Detect_System_root\models'
    }

    trainer = SVMTrainer(seed=42)

    # 数据加载和预处理
    X_train, y_train, X_test, y_test = trainer.load_and_preprocess_data(
        config['data_path'], config['features'], test_ratio=config['test_ratio']
    )

    # 选择是否进行过采样、欠采样或使用类别权重
    sample_method = "Random"  # 选择SMOTE、Undersample、None
    use_class_weight = False   # 是否使用类别权重

    if sample_method == "SMOTE":
        X_train_resampled, y_train_resampled = trainer.oversample_data(X_train, y_train, method="SMOTE")
    elif sample_method == "Random":
        X_train_resampled, y_train_resampled = trainer.oversample_data(X_train, y_train, method="Random")
    elif sample_method == "Undersample":
        X_train_resampled, y_train_resampled = trainer.undersample_data(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    # 使用类别权重
    if use_class_weight:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
        class_weight_dict = dict(zip(np.unique(y_train_resampled), class_weights))
        print(f"类别权重: {class_weight_dict}")
    else:
        class_weight_dict = None

    # 使用最佳参数训练模型
    trainer.train_with_best_params(X_train_resampled, y_train_resampled, class_weight=class_weight_dict, model_dir=config['model_dir'])

    # 在测试集上评估模型
    trainer.evaluate_on_test_set(X_test, y_test)

if __name__ == '__main__':
    main()
