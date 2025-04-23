import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

def extract_peak_segments(signal, peaks, window_size=1000):
    """提取峰值周围的数据段"""
    segments = []
    for peak in peaks:
        start = max(0, peak - window_size)
        end = min(len(signal), peak + window_size)
        
        if end - start == window_size * 2:
            segment = signal[start:end]
            segments.append(segment)
    return segments

def process_peaks_from_filtered_data(filtered_df, window_size=1000, threshold_factor=2):
    """
    处理过滤后的数据，提取峰值并返回峰值的个数以及峰值段。
    :param filtered_df: 经过过滤后的DataFrame
    :param window_size: 提取峰值段的窗口大小
    :param threshold_factor: 峰值的阈值系数，基于信号的均值和标准差
    :return: 峰值个数以及提取的峰值段
    """
    signal = filtered_df['Filtered_Data'].values
    
    # 输出信号的基本统计信息
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    print(f"信号统计: 均值={mean_signal:.4f}, 标准差={std_signal:.4f}")
    
    # 计算峰值的阈值
    threshold = mean_signal + threshold_factor * std_signal
    print(f"阈值: {threshold:.4f}")
    
    # 检测峰值
    peaks, _ = find_peaks(signal, height=threshold, distance=2000)
    print(f"检测到 {len(peaks)} 个峰值 (阈值: {threshold:.4f})")
    
    # 提取每个峰值周围的数据段
    segments = extract_peak_segments(signal, peaks, window_size)
    
    # 返回峰值个数和提取的峰值段
    return len(peaks), segments

