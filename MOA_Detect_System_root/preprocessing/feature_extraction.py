import numpy as np
import pandas as pd
import pywt
from scipy.fft import fft
from scipy.signal import hilbert
FEATURE_NAMES = [
    "peak_to_peak", "kurtosis", "skewness", "rms", "variance",
    "crest_factor", "shape_factor", "impulse_factor", "clearance_factor",
    "peak_index", "zero_crossings", "duration",
    "slope_mean", "abs_area", "max_amp", "mean_amp",
    "rise_time", "fall_time"
]

def extract_features_from_segments(peak_segments, fs=1000):
    """
    从多个峰值段中提取特征，并返回每个峰值段的特征信息
    :param peak_segments: 峰值段的列表，每个峰值段为一个信号数组
    :param fs: 信号的采样频率，默认1000Hz
    :return: 返回一个字典，键为峰值段的索引，值为每个峰值段提取的特征
    """
    features_dict = []  # 用于存储每个峰值段的特征
    # 遍历每个峰值段

    for idx, signal in enumerate(peak_segments):
        # 提取每个峰值段的特征
        features_dict.append(extract_features(signal, fs))

    return features_dict


def extract_features(signal, fs=1000):
    """提取单个信号段的特征"""
    # 时域特征
    peak_to_peak = np.ptp(signal)
    kurtosis = pd.Series(signal).kurtosis()
    skewness = pd.Series(signal).skew()
    std_dev = np.std(signal)
    variance = np.var(signal)
    max_value = np.max(signal)
    min_value = np.min(signal)
    mean_value = np.mean(signal)
    
    # 小波能量
    coeffs = pywt.wavedec(signal, 'db1', level=4)
    wavelet_energy = sum(np.sum(np.square(c)) for c in coeffs)
    
    # 主峰索引和主峰值
    peak_index = np.argmax(signal)
    peak_value = signal[peak_index]

    # 计算上升沿（从低值上升到主峰的索引间隔）
    start_index = np.where(signal > 0.1 * peak_value)[0][0]
    rise_time = peak_index - start_index if start_index < peak_index else 0

    # 计算下降沿（从主峰下降到低值的索引间隔）
    end_index = np.where(signal[peak_index:] < 0.1 * peak_value)[0]
    fall_time = end_index[0] if len(end_index) > 0 else 0

    # 频域特征
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    fft_values = np.abs(fft(signal))
    
    # 主要频率成分
    main_freq = freqs[np.argmax(fft_values)]
    
    # 频谱质心
    spectral_centroid = np.sum(freqs * fft_values) / np.sum(fft_values)
    
    # 频谱带宽
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_values) / np.sum(fft_values))
    
    # 频谱熵
    fft_values_norm = fft_values / np.sum(fft_values)
    spectral_entropy = -np.sum(fft_values_norm * np.log2(fft_values_norm + 1e-12))
    
    # 时频域特征
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
    mean_instantaneous_frequency = np.mean(instantaneous_frequency)
    
    # 短时能量
    frame_size = 256
    short_time_energy = np.array([np.sum(signal[i:i+frame_size]**2) for i in range(0, len(signal), frame_size)])
    mean_short_time_energy = np.mean(short_time_energy)
    
    # 小波熵
    wavelet_entropy = -np.sum(np.square(coeffs[-1]) * np.log(np.square(coeffs[-1]) + 1e-12))
    
    return {
        'peak_to_peak': peak_to_peak,
        'kurtosis': kurtosis,
        'skewness': skewness,
        'std_dev': std_dev,
        'variance': variance,
        'max_value': max_value,
        'min_value': min_value,
        'mean_value': mean_value,
        'wavelet_energy': wavelet_energy,
        'rise_time': rise_time,
        'fall_time': fall_time,
        'main_freq': main_freq,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_entropy': spectral_entropy,
        'mean_instantaneous_frequency': mean_instantaneous_frequency,
        'mean_short_time_energy': mean_short_time_energy,
        'wavelet_entropy': wavelet_entropy
    }

# 示例如何使用：
# 假设 peak_segments 是从峰值检测中获得的列表，每个元素为一个峰值段
# peak_segments = [segment1, segment2, ..., segmentN]

# 调用函数从多个峰值段提取特征
# features = extract_features_from_segments(peak_segments, fs=1000)

# 打印提取的特征
# for idx, feature_set in features.items():
#     print(f"峰值段 {idx} 的特征：")
#     print(feature_set)
