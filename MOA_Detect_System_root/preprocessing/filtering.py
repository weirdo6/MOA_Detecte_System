
import pandas as pd
import numpy as np
from pathlib import Path

def process_filtered_data(denoised_df, n, window_size):
    """
    过滤信号
    :param denoised_df: 小波降噪后的DataFrame
    :param n: 阈值系数
    :param window_size: 滑动窗口大小
    :return: 过滤后的DataFrame
    """
    # 假设传入的 DataFrame 包含 'Denoised_Data' 列
    data = denoised_df['Denoised_Data'].values.flatten()
    
    # 计算整体均值
    overall_mean = np.mean(np.abs(data))
    
    filtered_data = []
    for i in range(0, len(data), window_size):
        window = data[i:i + window_size]
        if len(window) == 0:
            continue
        # 过滤窗口内的数据
        filtered_window = window[np.abs(window) >= n * overall_mean]
        if len(filtered_window) > 0:
            filtered_data.append(filtered_window)

    if len(filtered_data) > 0:
        # 合并所有窗口的过滤数据
        filtered_data = np.concatenate(filtered_data)
        # 将结果转换为 DataFrame 返回
        filtered_df = pd.DataFrame({'Filtered_Data': filtered_data})
        return filtered_df
    else:
        # 如果没有数据被过滤掉，返回一个空的 DataFrame
        return pd.DataFrame({'Filtered_Data': []})


