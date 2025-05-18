import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

path = r"IMU_File\发正手后场高球\2025-05-14_12-10-15.txt"

data = pd.read_table(path, sep=',')
print(data)

# 原始数据读取
aX_data = data.iloc[:, 2].tolist()
aY_data = data.iloc[:, 3].tolist()
aZ_data = data.iloc[:, 4].tolist()

# 添加移动平均滤波
window_size = 10  # 滑动窗口大小
aX_filtered = np.convolve(aX_data, np.ones(window_size)/window_size, mode='same')
aY_filtered = np.convolve(aY_data, np.ones(window_size)/window_size, mode='same')
aZ_filtered = np.convolve(aZ_data, np.ones(window_size)/window_size, mode='same')

Gx_data = data.iloc[:, 10].tolist()
Gy_data = data.iloc[:, 11].tolist()
Gz_data = data.iloc[:, 12].tolist()

# 创建合并的DataFrame
imu_df = pd.DataFrame({
    'aX': aX_data,
    'aY': aY_data,
    'aZ': aZ_data,
    'Gx': Gx_data,
    'Gy': Gy_data,
    'Gz': Gz_data
})

# 保存到本地CSV文件
output_path = r"IMU_File\processed_imu_data.csv"
if os.path.exists(output_path):
    # 文件存在时追加数据，不包含表头
    imu_df.to_csv(output_path, mode='a', header=False, index=False)
else:
    # 文件不存在时创建新文件并写入数据
    imu_df.to_csv(output_path, index=False)
print(f"数据已保存到: {output_path}")

# 绘制滤波前后的加速度数据对比
plt.figure(figsize=(12, 8))
plt.subplot(2,1,1)
plt.plot(aX_data, label='aX_raw')
plt.plot(aY_data, label='aY_raw')
plt.plot(aZ_data, label='aZ_raw')
plt.title('Raw Acceleration Data')
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(aX_filtered, label='aX_filtered')
plt.plot(aY_filtered, label='aY_filtered')
plt.plot(aZ_filtered, label='aZ_filtered')
plt.title('Filtered Acceleration Data')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
print(aX_data)

# 绘制加速度数据图
plt.figure(figsize=(10, 6))
# plt.plot(aX_data, label='aX')
# plt.plot(aY_data, label='aY')
# plt.plot(aZ_data, label='aZ')
plt.plot(Gx_data, label='Gx')
plt.plot(Gy_data, label='Gy')
plt.plot(Gz_data, label='Gz')
plt.xlabel('Index')
plt.ylabel('Acceleration Value')
plt.title('Acceleration Data')
plt.legend()
plt.grid()
plt.show()