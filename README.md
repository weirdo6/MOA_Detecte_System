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
