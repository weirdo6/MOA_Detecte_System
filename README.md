# ZnO Surge Arrester Two-Stage Fault Diagnosis System

A Python-based framework for automated two-stage condition monitoring of zinc oxide (ZnO) surge arresters using partial discharge (PD) signal analysis. This repository implements:

1. **Stage 1 – Anomaly Detection**  
   - Binary classification (Normal vs. Abnormal)  
   - Logistic Regression with class-weighting to handle imbalance  
   - Features: spectral centroid, spectral bandwidth  

2. **Stage 2 – Defect Classification**  
   - Multi-class classification of abnormal signals into three defect types:  
     - Switch-box moisture  
     - Insulation degradation  
     - Other  
   - Support Vector Machine (RBF-kernel) with random oversampling  
   - Features: spectral bandwidth, spectral centroid, peak-to-peak, max/min, std. dev.  

3. **Preprocessing Pipeline**  
   - Wavelet-packet denoising  
   - Sliding-window energy filtering  
   - Peak-segment extraction  
   - Extraction of 18 time-, frequency- and time-frequency domain features  

---

## 📂 Repository Structure

```
.
├── preprocessing/                # Signal denoising, filtering, peak detection, feature extraction
│   ├── wavelet_denoising.py
│   ├── filtering.py
│   ├── peak_detection.py
│   └── feature_extraction.py
│
├── model_train/
│   ├── Anomaly_Detection_Stage/ # Stage 1 training & saved artifacts
│   │   └── LR_Final_Models.py
│   └── Defect_Categorization_Stage/ # Stage 2 training & saved artifacts
│       └── SVM_Final_Models.py
│
├── main.py                    # Two-stage inference pipeline
│
├── raw_data/                     # Example raw PD CSV files
│   └── *.csv
│
├── features_out/                 # Output folder for per-peak feature tables
│   └── *_features_by_peaks.csv
│
├── models/                       # Serialized scalers & trained models
│   ├── anomaly_scaler.pkl
│   ├── anomaly_model.pkl
│   ├── defect_scaler.pkl
│   └── defect_model.pkl
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## ⚙️ Prerequisites

- Python 3.8+  
- [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), [imbalanced-learn](https://imbalanced-learn.org/), [pywt](https://pywavelets.readthedocs.io/), [tqdm](https://github.com/tqdm/tqdm)  

Install via:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Preprocessing

Generate denoised, filtered, and per-peak feature tables:

```bash
python -m preprocessing.wavelet_denoising    # single-file or batch denoising
python -m preprocessing.filtering            # sliding-window filtering
python -m preprocessing.peak_detection       # peak-segment extraction
python -m preprocessing.feature_extraction   # multi-domain feature extraction
```

Or run the batch driver:

```bash
python main.py
```

This will:

1. Read raw PD CSVs from `raw_data/`
2. Save denoised → filtered → peak segments → feature tables to `features_out/`

### 2. Training

#### Stage 1: Anomaly Detection

```bash
python model_train/Anomaly_Detection_Stage/LR_Final_Models.py
```

- Trains Logistic Regression on extracted features  
- Saves `anomaly_scaler.pkl` & `anomaly_model.pkl` under `models/`

#### Stage 2: Defect Classification

```bash
python model_train/Defect_Categorization_Stage/SVM_Final_Models.py
```

- Trains RBF-kernel SVM on balanced peak features  
- Saves `defect_scaler.pkl` & `defect_model.pkl` under `models/`

### 3. Inference

Apply the full two-stage pipeline to new data:

```bash
python inference/main.py
```

- **Stage 1**: Scale & predict anomaly vs. normal  
- **Stage 2**: For predicted anomalies, scale & predict defect type  
- Prints per-peak diagnosis and overall status  

---

## 📊 Results

- **Anomaly Detection**  
  - Weighted Recall: 0.9763  Weighted F1: 0.9768  
  - Minor‐class recall: 0.9474  

- **Defect Classification**  
  - Overall Accuracy: 0.9267  
  - Class‐wise Recalls:  
    - Switch-box moisture (1.0000)  
    - Insulation degradation (0.8333)  
    - Other (0.7692)  

- **Single-Stage vs. Two-Stage**  
  - Single SVM on 4 classes:  
    - Accuracy: 0.8107  F1: 0.8263  
  - Two-stage system significantly outperforms single-stage, especially on minority classes.  

---

## 📖 References

1. Mallat S. “A Wavelet Tour of Signal Processing.” (3rd ed.). Academic Press, 2008.  
2. Chawla NV, et al. “SMOTE: Synthetic Minority Over-Sampling Technique.” J. AI Research, 2002.  
3. Pedregosa F, et al. “Scikit-learn: Machine Learning in Python.” JMLR, 2011.  

---

## 📝 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.  

Feel free to explore, reproduce the experiments, and extend the framework for your own industrial diagnostics scenarios!


# ZnO 避雷器两阶段故障诊断系统

基于局部放电（PD）信号分析的 ZnO（氧化锌）避雷器两阶段在线状态监测 Python 框架。本仓库功能包括：

1. **阶段一 — 异常诊断**  
   - 二分类（正常 vs 异常）  
   - 带类别权重的逻辑回归模型以应对数据不平衡  
   - 特征：频谱质心、频谱带宽  

2. **阶段二 — 缺陷分类**  
   - 对异常信号细分为三类缺陷：  
     - 开关盒受潮  
     - 绝缘劣化  
     - 其他类型  
   - RBF 核支持向量机（SVM）+ 随机过采样  
   - 特征：频谱带宽、频谱质心、峰峰值、最大/最小值、标准差  

3. **预处理管道**  
   - 小波包降噪  
   - 滑动窗口能量过滤  
   - 峰段提取  
   - 18 项时、频、时频域特征提取  

---

## 📂 仓库结构

```
.
├── preprocessing/                # 预处理脚本
│   ├── wavelet_denoising.py     # 小波去噪
│   ├── filtering.py             # 滑动窗口过滤
│   ├── peak_detection.py        # 峰段提取
│   └── feature_extraction.py    # 特征提取
│
├── model_train/
│   ├── Anomaly_Detection_Stage/      # 阶段一：训练与保存
│   │   └── LR_Final_Models.py
│   └── Defect_Categorization_Stage/  # 阶段二：训练与保存
│       └── SVM_Final_Models.py
│
├── inference/                    # 两阶段推理脚本
│   └── main.py
│
├── raw_data/                     # 示例原始 PD 信号
│   └── *.csv
│
├── features_out/                 # 保存各峰段特征表
│   └── *_features_by_peaks.csv
│
├── models/                       # 序列化模型与归一化器
│   ├── anomaly_scaler.pkl
│   ├── anomaly_model.pkl
│   ├── defect_scaler.pkl
│   └── defect_model.pkl
│
├── requirements.txt              # 依赖列表
└── README.md                     # 本文件
```

---

## ⚙️ 环境准备

- Python 3.8+  
- 必要库：NumPy、pandas、scikit-learn、imbalanced-learn、PyWavelets、tqdm  

```bash
pip install -r requirements.txt
```

---

## 🚀 使用指南

### 1. 数据预处理

批量生成去噪、过滤、峰段及特征表：

```bash
python -m preprocessing.wavelet_denoising    # 小波降噪
python -m preprocessing.filtering            # 滑动窗口过滤
python -m preprocessing.peak_detection       # 峰段提取
python -m preprocessing.feature_extraction   # 特征提取
```

或运行统一批处理脚本：

```bash
python inference/main.py
```

结果依次保存于 `features_out/`。

### 2. 模型训练

#### 阶段一 — 异常诊断

```bash
python model_train/Anomaly_Detection_Stage/LR_Final_Models.py
```

- 训练逻辑回归  
- 输出 `models/anomaly_scaler.pkl` & `models/anomaly_model.pkl`

#### 阶段二 — 缺陷分类

```bash
python model_train/Defect_Categorization_Stage/SVM_Final_Models.py
```

- 训练 RBF-kernel SVM  
- 输出 `models/defect_scaler.pkl` & `models/defect_model.pkl`

### 3. 两阶段推理

```bash
python inference/main.py
```

- **阶段一**：对新信号进行归一化 & 异常/正常预测  
- **阶段二**：若判定为“异常”，则再归一化 & 细分三类缺陷  

---

## 📊 实验结果

- **异常诊断**  
  - 加权召回：97.63%  加权 F1：97.68%  
  - 少数类召回：94.74%  

- **缺陷分类**  
  - 总体准确率：92.67%  
  - 关键类别召回：  
    - 开关盒受潮：100%  
    - 绝缘劣化：83.33%  
    - 其他类型：76.92%  

- **单阶段 vs. 两阶段**  
  - 单阶段 SVM 四分类准确率：81.07%  F1：82.63%  
  - 两阶段系统在少数类和整体性能上均有显著提升  

---

## 📖 参考文献

1. Mallat S. *A Wavelet Tour of Signal Processing.* Academic Press, 2008.  
2. Chawla NV, et al. “SMOTE: Synthetic Minority Over-Sampling Technique.” *J. AI Research*, 2002.  
3. Pedregosa F, et al. “Scikit-learn: Machine Learning in Python.” *JMLR*, 2011.  

---

## 📝 许可证

本项目采用 **MIT 许可证**，详见 [LICENSE](LICENSE)。  

欢迎使用、复现与扩展，用于更多工业故障在线监测场景！
