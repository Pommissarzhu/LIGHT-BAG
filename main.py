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

# 滑动窗口处理
for i in range(num_windows):
    start_idx = i * stride
    end_idx = start_idx + window_size
    window_data = data.iloc[start_idx:end_idx]
    
    # 创建可视化图表
    plt.figure(figsize=(12, 8))
    
    # 绘制加速度数据
    plt.subplot(2, 1, 1)
    plt.plot(window_data['aX'], label='aX')
    plt.plot(window_data['aY'], label='aY')
    plt.plot(window_data['aZ'], label='aZ')
    plt.ylim(-200, 200)  # 设置加速度数据固定显示范围
    plt.title(f'Window {i+1} - Acceleration Data')
    plt.legend()
    plt.grid()
    
    # 绘制陀螺仪数据
    plt.subplot(2, 1, 2)
    plt.plot(window_data['Gx'], label='Gx')
    plt.plot(window_data['Gy'], label='Gy')
    plt.plot(window_data['Gz'], label='Gz')
    plt.ylim(-1500, 2500)  # 设置陀螺仪数据固定显示范围
    plt.title(f'Window {i+1} - Gyroscope Data')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f"{i+1}.png")
    plt.savefig(output_path)
    plt.close()
    
print(f"已完成所有窗口的可视化，共保存{num_windows}张图片到{output_dir}目录")