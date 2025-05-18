import random
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

def create_serve_datasets(data_dir, test_ratio=0.2, random_seed=42):  # 移除val_ratio参数
    # 1. 定义标签映射（根据文件名生成）
    label_map = {
        'backhand_short_serve': 0,
        'forehand_short_serve': 1,
        'forehand_deep_high_serve': 2
    }

    # 2. 加载所有数据和标签（原有代码不变）
    all_data = []
    all_labels = []
    total_adjusted = 0  # 总调整样本数
    
    for label_name in label_map.keys():
        file_path = f"{data_dir}/{label_name}.npy"
        data = np.load(file_path)  # 形状 [30, 32, 6]
        label_adjusted = 0  # 当前标签调整的样本数
        
        # 新增预处理：调整第1、2、4、5列（索引0、1、3、4）的符号
        for i in range(data.shape[0]):  # 遍历每个样本
            sample = data[i]  # 单个样本形状 [32, 6]
            cols_to_check = [0]  # 对应第1、2、4、5列（索引从0开始）
            
            # 检查是否需要翻转符号
            need_flip = False
            for col in cols_to_check:
                column_values = sample[:, col]
                max_val = np.max(column_values)
                max_abs_val = np.max(np.abs(column_values))
                # 允许微小浮点误差（使用np.isclose判断）
                if not np.isclose(max_val, max_abs_val, atol=1e-6):
                    need_flip = True
                    break
            
            # 若需要则翻转这4列的符号
            if need_flip:
                for col in [0, 1, 3, 4]:
                    sample[:, col] *= -1
                data[i] = sample  # 更新当前样本数据
                label_adjusted += 1  # 当前标签调整数+1
        
        # 统计输出
        print(f"标签 {label_name} 调整符号的样本数: {label_adjusted}/{data.shape[0]}")
        total_adjusted += label_adjusted  # 累加到总调整数
        
        # 转换为PyTorch张量并添加到列表
        all_data.append(torch.tensor(data, dtype=torch.float32))
        # 添加对应标签（重复30次）
        all_labels.extend([label_map[label_name]] * data.shape[0])
    
    # 输出总调整数
    print(f"\n所有标签总计调整符号的样本数: {total_adjusted}/{len(all_data)*30}")
    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    # 3. 分层划分训练集（80%）和测试集（20%）
    from sklearn.model_selection import StratifiedShuffleSplit
    all_labels_np = all_labels.numpy()
    
    # 仅划分训练集和测试集（20%测试）
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
    train_idx, test_idx = next(sss_test.split(all_data, all_labels_np))

    # 提取训练集和测试集数据
    train_data = list(zip(all_data[train_idx], all_labels[train_idx]))
    test_data = list(zip(all_data[test_idx], all_labels[test_idx]))

    return train_data, test_data  # 返回原始数据（未归一化，交叉验证中逐折处理）

def print_class_distribution(dataset, name):
        labels = [label.item() for _, label in dataset]
        counts = {0:0, 1:0, 2:0}
        for l in labels:
            counts[l] += 1
        print(f"{name}类别分布: 0类={counts[0]}, 1类={counts[1]}, 2类={counts[2]}")

def set_random_seed(seed: int = 42, deterministic: bool = True):
    """
    统一设置随机数种子，确保实验可复现
    
    Args:
        seed (int): 基础随机种子（默认42）
        deterministic (bool): 是否启用完全确定性模式（关闭非确定性优化）
    Returns:
        int: 实际使用的随机种子（方便外部组件复用）
    """
    # Python内置random模块
    random.seed(seed)
    # numpy随机数
    np.random.seed(seed)
    # PyTorch CPU随机数
    torch.manual_seed(seed)
    # PyTorch 所有GPU随机数（如果有）
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 关闭CuDNN非确定性优化（针对NVIDIA GPU）
    if deterministic and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True  # 强制使用确定性算法
        torch.backends.cudnn.benchmark = False      # 关闭自动选择最优算法
    
    # 关闭MPS非确定性（针对Apple GPU）
    if deterministic and torch.backends.mps.is_available():
        torch.backends.mps.deterministic = True     # MPS确定性模式

# 在main函数最开始调用（示例）
if __name__ == "__main__":
    # 优先设置随机种子（确保所有操作可复现）
    fixed_seed = set_random_seed(seed=352, deterministic=False)  # 获取统一种子
    
    DATA_DIR = "/Users/aero/LIGHT-BAG/dataset"
    
    # 获取原始训练集（80%）和测试集（20%）
    train_data_raw, test_data_raw = create_serve_datasets(DATA_DIR, random_seed=42)

    # 5折交叉验证设置（使用统一种子）
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=fixed_seed)  # 复用统一种子
    fold_results = []  # 保存各折验证结果

    # 提取训练集的特征和标签（用于生成折叠索引）
    train_features_raw = torch.stack([item[0] for item in train_data_raw])
    train_labels_raw = torch.stack([item[1] for item in train_data_raw])

    # 保存每一折的scaler（在交叉验证循环中新增）
    scalers = []  # 新增：保存各折的归一化器

    # 5折交叉验证设置（修改原循环，添加scaler保存）
    for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(skf.split(train_features_raw, train_labels_raw)):
        print(f"\n===== 第 {fold_idx+1} 折训练 =====")
        
        # 划分当前折叠的训练/验证数据
        train_fold_data = [train_data_raw[i] for i in train_fold_idx]
        val_fold_data = [train_data_raw[i] for i in val_fold_idx]

        # 逐折计算归一化参数（仅用当前折叠的训练数据）
        train_fold_features = torch.stack([item[0] for item in train_fold_data])
        flattened_train = train_fold_features.reshape(-1, 6).numpy()
        scaler = MinMaxScaler()
        scaler.fit(flattened_train)
        scalers.append(scaler)  # 新增：保存当前折的scaler

        # 定义归一化函数（当前折叠专用）
        def normalize(tensor):
            flattened = tensor.reshape(-1, 6).numpy()
            normalized = scaler.transform(flattened)
            return torch.tensor(normalized.reshape(tensor.shape), dtype=torch.float32)

        # 归一化当前折叠的训练/验证数据
        train_fold_normalized = [
            (normalize(data), label) for (data, label) in train_fold_data
        ]
        val_fold_normalized = [
            (normalize(data), label) for (data, label) in val_fold_data
        ]

        # 转换为Dataset对象
        train_fold_ds = ServeDataset(
            data=torch.stack([item[0] for item in train_fold_normalized]),
            labels=torch.stack([item[1] for item in train_fold_normalized])
        )
        val_fold_ds = ServeDataset(
            data=torch.stack([item[0] for item in val_fold_normalized]),
            labels=torch.stack([item[1] for item in val_fold_normalized])
        )

        # 创建数据加载器（batch_size=16）
        train_loader = DataLoader(train_fold_ds, batch_size=16, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_fold_ds, batch_size=16, shuffle=False, num_workers=8)

        # 模型初始化（与原代码一致）
        input_size = 6
        hidden_size = 64
        num_layers = 2
        num_classes = 3
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMCategorizer(input_size, hidden_size, num_layers, num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # 单折训练（50 epoch）
        best_val_acc = 0.0
        for epoch in range(50):
            # 训练模式（与原代码一致）
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (data, labels) in enumerate(train_loader):
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            train_loss /= len(train_loader)
            train_acc = 100 * correct / total

            # 验证模式（与原代码一致）
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
            val_loss /= len(val_loader)
            val_acc = 100 * correct_val / total_val

            # 学习率衰减和日志（与原代码一致）
            scheduler.step()
            print(f"Epoch [{epoch+1}/50], Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"best_lstm_model_fold{fold_idx+1}.pth")

        fold_results.append(best_val_acc)
        print(f"第 {fold_idx+1} 折最佳验证准确率: {best_val_acc:.2f}%")

    # 输出交叉验证结果
    avg_val_acc = sum(fold_results) / len(fold_results)
    print(f"\n5折交叉验证平均准确率: {avg_val_acc:.2f}%")

    # 最终测试集评估（使用任意一折的最佳模型或集成模型，此处示例使用第1折模型）
    # model.load_state_dict(torch.load("best_lstm_model_fold1.pth"))
    # model.eval()
    # test_correct = 0
    # test_total = 0
    # 归一化测试集（使用第1折的scaler，实际应使用所有训练数据的scaler，此处简化）
    # test_features = torch.stack([item[0] for item in test_data_raw])
    # test_labels = torch.stack([item[1] for item in test_data_raw])
    # test_normalized = normalize(test_features)  # 使用第1折的normalize函数
    # test_ds = ServeDataset(data=test_normalized, labels=test_labels)
    # test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=8)
    
    # with torch.no_grad():
    #     for data, labels in test_loader:
    #         data, labels = data.to(device), labels.to(device)
    #         outputs = model(data)
    #         _, predicted = torch.max(outputs.data, 1)
    #         test_total += labels.size(0)
    #         test_correct += (predicted == labels).sum().item()
    # print(f"最终测试集准确率: {100 * test_correct / test_total:.2f}%")

    # 最终测试集评估（修改为遍历所有折模型）
    fold_test_accs = []  # 保存各折测试准确率
    
    for fold_idx in range(5):
        # 加载当前折的最佳模型
        model_path = f"best_lstm_model_fold{fold_idx+1}.pth"
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # 使用当前折的scaler归一化测试集
        scaler = scalers[fold_idx]
        def normalize_current(tensor):
            flattened = tensor.reshape(-1, 6).numpy()
            normalized = scaler.transform(flattened)
            return torch.tensor(normalized.reshape(tensor.shape), dtype=torch.float32)

        # 准备测试数据
        test_features = torch.stack([item[0] for item in test_data_raw])
        test_labels = torch.stack([item[1] for item in test_data_raw])
        test_normalized = normalize_current(test_features)
        test_ds = ServeDataset(data=test_normalized, labels=test_labels)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=8)

        # 评估当前折模型
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_acc = 100 * test_correct / test_total
        fold_test_accs.append(test_acc)
        print(f"第 {fold_idx+1} 折模型测试准确率: {test_acc:.2f}%")

    # 计算平均测试准确率
    avg_test_acc = sum(fold_test_accs) / len(fold_test_accs)
    print(f"\n5折模型平均测试准确率: {avg_test_acc:.2f}%")