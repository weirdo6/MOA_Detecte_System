from preprocessing import wavelet_denoising, filtering, peak_detection, feature_extraction
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# 屏蔽 sklearn 模型反序列化的版本不一致警告
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
# 屏蔽 “X does not have valid feature names” 警告
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# —— 在此处填写各阶段所需特征 —— 
ANOMALY_FEATURES = ['spectral_centroid', 'spectral_bandwidth']
DEFECT_FEATURES  = ['spectral_bandwidth', 'spectral_centroid', 'max_value', 
                    'peak_to_peak', 'std_dev', 'min_value']

# —— 在此处填写已保存的模型和 scaler 路径 —— 
ANOMALY_SCALER_PATH = Path("MOA_Detect_System_root/models/anomaly_scaler.pkl")
ANOMALY_MODEL_PATH  = Path("MOA_Detect_System_root/models/anomaly_model.pkl")
DEFECT_SCALER_PATH  = Path("MOA_Detect_System_root/models/defect_scaler.pkl")
DEFECT_MODEL_PATH   = Path("MOA_Detect_System_root/models/defect_model.pkl")

def classify_peaks(features_df: pd.DataFrame):
    """
    对单个文件的各峰段特征做两阶段识别：
    1) 异常检测
    2) 若异常则缺陷分类
    返回：dict {peak_index: ('正常' or defect_label)}
    """
    # 加载模型
    anomaly_scaler = joblib.load(ANOMALY_SCALER_PATH)
    anomaly_model  = joblib.load(ANOMALY_MODEL_PATH)
    defect_scaler  = joblib.load(DEFECT_SCALER_PATH)
    defect_model   = joblib.load(DEFECT_MODEL_PATH)

    results = {}
    for peak_col in features_df.columns:
        # 准备第一阶段向量
        try:
            x_anom = features_df.loc[ANOMALY_FEATURES, peak_col].values.reshape(1, -1)
        except KeyError as e:
            raise KeyError(f"异常诊断阶段缺少特征: {e}")
        x_anom_scaled = anomaly_scaler.transform(x_anom)
        is_anomaly = anomaly_model.predict(x_anom_scaled)[0]  # 0=normal,1=anomaly

        if is_anomaly == 0:
            results[peak_col] = '正常'
            print(f"  峰段 {peak_col}: 正常")
            continue

        # 第二阶段：缺陷分类
        try:
            print(f"  峰段 {peak_col}: 缺陷")
            x_def = features_df.loc[DEFECT_FEATURES, peak_col].values.reshape(1, -1)
        except KeyError as e:
            raise KeyError(f"缺陷分类阶段缺少特征: {e}")
        x_def_scaled = defect_scaler.transform(x_def)
        label = defect_model.predict(x_def_scaled)[0]
        # 映射数值标签到中文描述（假设0:开关盒受潮,1:绝缘受损,2:其他类型）
        label_map = {0: '开关盒受潮', 1: '绝缘受损', 2: '其他类型'}
        results[peak_col] = label_map.get(label, str(label))
    return results

def process_signal_file(input_file: Path, output_dir: Path):
    print(f"\n>>> 正在处理文件: {input_file.name}")

    # 1) 小波去噪
    denoised_df = wavelet_denoising.process_normal_data(input_file)
    # 2) 信号过滤
    filtered_df = filtering.process_filtered_data(denoised_df, n=4, window_size=10000)
    # 3) 峰段提取
    _, segments = peak_detection.process_peaks_from_filtered_data(filtered_df)
    print(f"    峰段数: {len(segments)}")
    # 4) 特征提取
    features = feature_extraction.extract_features_from_segments(segments)
    # 将 List[dict] 转为 DataFrame（行=特征, 列=峰段编号）
    feature_names = list(features[0].keys())
    mat = [list(f.values()) for f in features]
    features_df = pd.DataFrame(mat).T
    features_df.index = feature_names
    features_df.columns = [str(i) for i in range(len(features))]

    # 保存特征表
    output_csv = output_dir / f"{input_file.stem}_features_by_peaks.csv"
    features_df.to_csv(output_csv, index=True)
    print(f"✅ 特征表已保存至: {output_csv}")

    # 5) 两阶段分类
    classification = classify_peaks(features_df)
    print("    两阶段识别结果：")
    for peak, label in classification.items():
        print(f"      峰段 {peak}: {label}")

def batch_process(input_dir: Path, output_dir: Path):
    print(f">>> 批处理开始：遍历文件夹 {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_dir.glob("*.csv"):
        try:
            process_signal_file(input_file, output_dir)
        except Exception as e:
            print(f"❌ 文件处理失败: {input_file.name}，原因: {e}")

    print("\n>>> 所有文件处理完成 ✅")

def main():
    input_dir  = Path(r"MOA_Detect_System_root/raw_data")
    output_dir = Path(r"MOA_Detect_System_root/features_out")
    batch_process(input_dir, output_dir)

if __name__ == "__main__":
    main()
