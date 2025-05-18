import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler


class ServeDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data (Tensor): 形状为 [样本数, 时间步, 特征数] 的数据张量
            labels (Tensor): 形状为 [样本数] 的标签张量（整数编码）
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class LSTMCategorizer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(LSTMCategorizer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层（batch_first=True适配输入形状 [batch, seq_len, input_size]）
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # 多层LSTM时添加dropout
        )
        
        # 全连接分类头
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x形状: [batch_size, seq_len, input_size] = [8, 32, 6]（示例）
        
        # 初始化隐藏状态和细胞状态（默认使用0初始化）
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播，获取最后一个时间步的隐藏状态
        out, _ = self.lstm(x, (h0, c0))  # out形状: [batch_size, seq_len, hidden_size]
        
        # 取最后一个时间步的特征（适合时序分类任务）
        out = out[:, -1, :]  # 形状: [batch_size, hidden_size]
        
        # 全连接分类
        out = self.fc(out)  # 形状: [batch_size, num_classes]
        return out

def create_serve_datasets(data_dir, val_ratio=0.2, test_ratio=0.1, random_seed=42):
    # 1. 定义标签映射（根据文件名生成）
    label_map = {
        'backhand_short_serve': 0,
        'forehand_short_serve': 1,
        'forehand_deep_high_serve': 2
    }

    # 2. 加载所有数据和标签
    all_data = []
    all_labels = []
    for label_name in label_map.keys():
        # 加载npy文件（假设文件路径正确）
        file_path = f"{data_dir}/{label_name}.npy"
        data = np.load(file_path)  # 形状 [30, 32, 6]
        # 转换为PyTorch张量并添加到列表
        all_data.append(torch.tensor(data, dtype=torch.float32))
        # 添加对应标签（重复30次）
        all_labels.extend([label_map[label_name]] * data.shape[0])

    # 合并所有数据（形状 [90, 32, 6]）
    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    # 3. 划分训练集/验证集/测试集
    total_samples = len(all_data)
    test_size = int(total_samples * test_ratio)
    val_size = int(total_samples * val_ratio)
    train_size = total_samples - val_size - test_size

    # 固定随机种子保证可复现
    generator = torch.Generator().manual_seed(random_seed)
    train_data, val_data, test_data = random_split(
        dataset=list(zip(all_data, all_labels)),
        lengths=[train_size, val_size, test_size],
        generator=generator
    )

    # 4. 训练集Min-Max归一化（仅用训练集计算统计量）
    # 提取训练集数据（形状 [train_size, 32, 6]）
    train_features = torch.stack([item[0] for item in train_data])
    # 展平时间步维度（形状 [train_size*32, 6]）
    flattened_train = train_features.reshape(-1, 6)
    
    # 创建并拟合归一化器
    scaler = MinMaxScaler()
    scaler.fit(flattened_train.numpy())  # 用训练集拟合

    # 定义归一化函数（应用到所有数据集）
    def normalize(tensor):
        flattened = tensor.reshape(-1, 6)
        normalized = scaler.transform(flattened)
        return torch.tensor(normalized.reshape(tensor.shape), dtype=torch.float32)

    # 对各数据集应用归一化（注意：验证集/测试集使用训练集的scaler）
    train_data_normalized = [
        (normalize(data), label) for (data, label) in train_data
    ]
    val_data_normalized = [
        (normalize(data), label) for (data, label) in val_data
    ]
    test_data_normalized = [
        (normalize(data), label) for (data, label) in test_data
    ]

    # 5. 转换为Dataset对象
    train_dataset = ServeDataset(
        data=torch.stack([item[0] for item in train_data_normalized]),
        labels=torch.stack([item[1] for item in train_data_normalized])
    )
    val_dataset = ServeDataset(
        data=torch.stack([item[0] for item in val_data_normalized]),
        labels=torch.stack([item[1] for item in val_data_normalized])
    )
    test_dataset = ServeDataset(
        data=torch.stack([item[0] for item in test_data_normalized]),
        labels=torch.stack([item[1] for item in test_data_normalized])
    )

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    # 数据目录（根据实际路径调整）
    DATA_DIR = "/Users/aero/LIGHT-BAG/dataset"
    
    # 创建数据集
    train_ds, val_ds, test_ds = create_serve_datasets(DATA_DIR)

    # 查看数据形状验证
    print(f"训练集样本数: {len(train_ds)}，数据形状: {train_ds[0][0].shape}")  # 应输出 (32, 6)
    print(f"验证集样本数: {len(val_ds)}")
    print(f"测试集样本数: {len(test_ds)}")

    # 创建数据加载器（调整batch_size为8，可根据GPU内存调整）
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)

    # 模型初始化（超参数根据经验设置）
    input_size = 6  # 特征维度
    hidden_size = 64  # LSTM隐藏层大小
    num_layers = 2  # LSTM层数
    num_classes = 3  # 分类类别数
    # 设备检测：优先MPS（Apple GPU）→ CUDA（NVIDIA GPU）→ CPU
    device = torch.device(
        "mps" if torch.backends.mps.is_available() 
        else "cuda" if torch.cuda.is_available() 
        else "cpu"
    )
    model = LSTMCategorizer(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes
    ).to(device)

    # 损失函数和优化器（经验设置）
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率衰减

    # 训练超参数
    num_epochs = 50

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)  # 数据移至GPU
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计指标
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算训练指标
        train_loss /= len(train_loader)
        train_acc = 100 * correct / total

        # 验证模式
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # 计算验证指标
        val_loss /= len(val_loader)
        val_acc = 100 * correct_val / total_val

        # 学习率衰减
        scheduler.step()

        # 打印epoch结果
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_lstm_model.pth")
            print(f"保存最佳模型（验证准确率: {val_acc:.2f}%）")

    # 测试集评估（使用最佳模型）
    model.load_state_dict(torch.load("best_lstm_model.pth"))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    print(f"\n测试集准确率: {100 * test_correct / test_total:.2f}%")

    # 打印前5个训练集样本的具体数据和标签（原有代码保留）
    print("\n前5个训练集样本详情：")
    for i in range(5):
        sample_data, sample_label = train_ds[i]
        print(f"样本{i+1}:")
        print(f"  数据形状: {sample_data.shape}")
        print(f"  数据前2时间步: {sample_data[:2].numpy()}")  # 显示前2个时间步的特征值
        print(f"  标签: {sample_label.item()}\n")