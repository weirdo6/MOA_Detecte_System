import numpy as np
import pandas as pd
import pywt
from pathlib import Path
from tqdm import tqdm

# ----- 小波去噪核心函数 -----
def calc_entropy(wpt, level, nd):
    ent = []
    for i in range(2 ** level):
        c = wpt[nd[i]].data
        sq = np.square(c)
        ent.append(-np.sum(sq * np.log(sq + 1e-10)))
    return ent

def wden(data, level, wavelet):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # 估计每层噪声 sigma
    sigmas = [np.median(np.abs(c)) / 0.6745 for c in coeffs[1:]]
    # 对细节系数做软阈值
    for i in range(1, len(coeffs)):
        thr = (0.3936 + 0.1829 * np.log2(len(data))) * sigmas[i-1]
        coeffs[i] = pywt.threshold(coeffs[i], thr, mode='soft')
    return pywt.waverec(coeffs, wavelet)

def denoise_chunk(chunk):
    # 小波包分解
    wpt = pywt.WaveletPacket(data=chunk, wavelet='db8', mode='symmetric', maxlevel=6)
    nd = [node.path for node in wpt.get_leaf_nodes(True)]
    ent = calc_entropy(wpt, 6, nd)
    # 把熵最大的子带置零
    wpt[nd[int(np.argmax(np.abs(ent)))]] .data = np.zeros_like(wpt[nd[0]].data)
    # 重构一次
    data1 = wpt.reconstruct(update=False)
    # 再用 wden 进一步去噪
    return wden(data1, 7, 'db7')

# ----- 主处理函数 -----
def process_normal_data(input_file: Path, chunk_size: int = 100_000) -> pd.DataFrame:
    """
    读取 CSV 中的 'Original_Data' 列，分块做波形去噪，自动 pad/trim，返回合并后的 DataFrame。
    """
    # 1. 读入并准备
    df = pd.read_csv(input_file, header=0, low_memory=False)
    df.columns = ['Original_Data']
    df['Original_Data'] = pd.to_numeric(df['Original_Data'], errors='coerce').fillna(0.0)
    data = df['Original_Data'].values
    total_len = len(data)

    # 2. 分块处理
    denoised_parts = []
    for start in tqdm(range(0, total_len, chunk_size), desc="去噪进度"):
        end = min(start + chunk_size, total_len)
        chunk = data[start:end]

        # 如果末尾长度不足，前填充到 chunk_size
        orig_len = len(chunk)
        if orig_len < chunk_size:
            pad = np.zeros(chunk_size - orig_len, dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad])

        # 去噪
        den = denoise_chunk(chunk)

        # 截回到原始长度
        denoised_parts.append(den[:orig_len])

    # 3. 合并并返回
    denoised = np.concatenate(denoised_parts)
    return pd.DataFrame({'Denoised_Data': denoised})

# ----- 测试运行 -----
if __name__ == "__main__":
    input_file = Path("MOA_Detect_System_root/raw_data/202412161050.csv")
    denoised_df = process_normal_data(input_file)
    print(f"去噪结束，共输出 {len(denoised_df)} 条数据。")
    print(denoised_df.head())
