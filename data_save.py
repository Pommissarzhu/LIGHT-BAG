import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

path = r"IMU_File\processed_imu_data.csv"
data = pd.read_csv(path)

# 创建输出目录
output_dir = r"IMU_File\window_plots"
os.makedirs(output_dir, exist_ok=True)

# 设置窗口参数
window_size = 32
stride = 4
total_samples = len(data)
num_windows = (total_samples - window_size) // stride + 1

anno = [80, 120, 161, 205, 246, 289, 328, 370, 411, 452]
for i in range(len(anno)):  # 修改为使用列表长度
    anno[i] = anno[i] - 1
print(anno)

selected_windows = []  # 用于存储选中的窗口数据

# 滑动窗口处理
for i in range(num_windows):
    start_idx = i * stride
    end_idx = start_idx + window_size
    window_data = data.iloc[start_idx:end_idx]

    if i in anno:
        selected_windows.append(window_data.to_numpy())  # 转换为numpy数组去掉表头

# 将整个selected_windows转换为numpy数组
selected_windows = np.array(selected_windows)
print(f"selected_windows的最终形状: {selected_windows.shape}")

# 保存为npy文件
np.save(os.path.join(output_dir, 'selected_windows.npy'), selected_windows)
print(f"已保存numpy数组到: {os.path.join(output_dir, 'selected_windows.npy')}")

# 打印每个窗口数据的维度
print(f"共选中了{len(selected_windows)}个窗口")
for idx, window in enumerate(selected_windows):
    print(f"窗口 {idx+1} 的维度: {window.shape}")  # 这里window是numpy数组，有shape属性

# print(selected_windows[0])