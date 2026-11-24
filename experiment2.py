import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 设置随机种子，保证每次运行结果一致
torch.manual_seed(42)

print("="*30, "实验 1: 4维张量创建与归一化", "="*30)
# 1. 创建形状为 (batch_size, 3, height, width) 的随机 Tensor
batch_size, channels, height, width = 4, 3, 32, 32
images = torch.randn(batch_size, channels, height, width)

# 2. 计算每个通道的均值和标准差 (dim=(0, 2, 3) 表示在 batch, height, width 维度上求，保留 channel 维度)
# keepdim=True 保持维度，方便后续广播计算
mean = images.mean(dim=(0, 2, 3), keepdim=True)
std = images.std(dim=(0, 2, 3), keepdim=True)

# 3. 对每个图像进行归一化处理 (加上 1e-8 防止标准差为0导致除零错误)
normalized_images = (images - mean) / (std + 1e-8)

print(f"原始张量形状: {images.shape}")
print(f"通道均值形状: {mean.shape}")
print(f"归一化后形状: {normalized_images.shape}")
# 验证归一化效果（均值应接近0，标准差应接近1）
print(f"归一化后通道均值: {normalized_images.mean(dim=(0, 2, 3))}")
print(f"归一化后通道标准差: {normalized_images.std(dim=(0, 2, 3))}")


print("\n" + "="*30, "实验 2: 损失函数定义与正确性验证", "="*30)
# 模拟真实值和预测值
y_true_reg = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_pred_reg = torch.tensor([1.1, 1.9, 3.2, 3.8])

# 1. 均方差损失 (MSE)
mse_manual = ((y_true_reg - y_pred_reg) ** 2).mean()
mse_torch = nn.MSELoss()(y_pred_reg, y_true_reg)
print(f"[MSE] 手动计算: {mse_manual:.4f}, PyTorch内置: {mse_torch.item():.4f} -> {'一致' if torch.isclose(mse_manual, mse_torch) else '不一致'}")

# 2. 平均绝对误差损失 (MAE)
mae_manual = torch.abs(y_true_reg - y_pred_reg).mean()
mae_torch = nn.L1Loss()(y_pred_reg, y_true_reg)
print(f"[MAE] 手动计算: {mae_manual:.4f}, PyTorch内置: {mae_torch.item():.4f} -> {'一致' if torch.isclose(mae_manual, mae_torch) else '不一致'}")

# 3. 多分类交叉损失 (CrossEntropy Loss)
# 注意：PyTorch的CrossEntropyLoss接收的是未经过Softmax的logits，target是类别索引(0, 1, 2...)
logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
targets = torch.tensor([0, 1])
ce_torch = nn.CrossEntropyLoss()(logits, targets)
# 手动实现验证
softmax_logits = F.softmax(logits, dim=1)
log_probs = torch.log(softmax_logits)
nll_manual = -log_probs[range(len(targets)), targets].mean()
print(f"[CrossEntropy] 手动计算: {nll_manual:.4f}, PyTorch内置: {ce_torch.item():.4f} -> {'一致' if torch.isclose(nll_manual, ce_torch) else '不一致'}")

# 4. 二分类交叉损失 (BCELoss)
# 注意：BCELoss要求输入已经经过Sigmoid（在0-1之间）
bce_pred = torch.tensor([0.8, 0.2, 0.6, 0.9])
bce_true = torch.tensor([1.0, 0.0, 1.0, 1.0])
bce_torch = nn.BCELoss()(bce_pred, bce_true)
# 手动实现验证
bce_manual = -(bce_true * torch.log(bce_pred) + (1 - bce_true) * torch.log(1 - bce_pred)).mean()
print(f"[BCELoss] 手动计算: {bce_manual:.4f}, PyTorch内置: {bce_torch.item():.4f} -> {'一致' if torch.isclose(bce_manual, bce_torch) else '不一致'}")


print("\n" + "="*30, "实验 3: 常用张量操作 (对应实验目的)", "="*30)
# 1. Tensor 与 Numpy 转换
np_array = np.array([[1, 2], [3, 4]])
tensor_from_np = torch.from_numpy(np_array)
print(f"1. Numpy转Tensor:\n{tensor_from_np}")

# 2. 索引与修改
t = torch.tensor([[1, 2], [3, 4]])
t[0, 1] = 99
print(f"2. 修改索引后的Tensor:\n{t}")

# 3. 广播机制
a = torch.tensor([[1], [2]])  # 形状 (2, 1)
b = torch.tensor([10, 20])    # 形状 (2,)
print(f"3. 广播相加 (2,1)+(2,):\n{a + b}")

# 4. 归并操作
print(f"4. 全部元素求和: {t.sum()}")
print(f"   按列求和: {t.sum(dim=0)}")

# 5. 比较操作
print(f"5. 元素是否大于2:\n{t > 2}")

# 6. 矩阵操作
m1 = torch.tensor([[1, 2], [3, 4]])
m2 = torch.tensor([[2, 0], [1, 2]])
print(f"6. 矩阵乘法:\n{torch.matmul(m1, m2)}")