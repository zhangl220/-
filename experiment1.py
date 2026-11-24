import numpy as np

# ============================================================
# 1. 生成模拟成绩数据，保存为 scores.csv
# ============================================================
np.random.seed(2024)
n = 500

math = np.random.normal(72, 8, n)
chinese = np.random.normal(68, 10, n)
english = np.random.normal(75, 7, n)

# 截断到 [0, 100] 并取整
math = np.clip(math, 0, 100).astype(int)
chinese = np.clip(chinese, 0, 100).astype(int)
english = np.clip(english, 0, 100).astype(int)

# 合并为二维数组：500行，3列
scores = np.column_stack((math, chinese, english))

# 保存为 CSV 文件
np.savetxt(
    "scores.csv",
    scores,
    delimiter=",",
    fmt="%d",
    header="数学,语文,英语",
    comments="",
    encoding="utf-8"
)

print("数据已生成，保存为 scores.csv")
print("前5行数据预览：")
print(scores[:5])

# ============================================================
# 2. 步骤1：加载数据
# ============================================================
data = np.loadtxt(
    "scores.csv",
    delimiter=",",
    skiprows=1,
    dtype=int,
    encoding="utf-8"
)

subjects = ["数学", "语文", "英语"]

print("\n数据形状（行数，列数）：", data.shape)
print("前5行数据：\n", data[:5])

# ============================================================
# 3. 步骤2：计算各科平均分和方差
# ============================================================
print("\n【各科成绩统计】")
print(f"{'科目':<6}{'平均分':>8}{'方差':>10}")

for i, sub in enumerate(subjects):
    col = data[:, i]              # 提取第 i 列
    mean_val = np.mean(col)       # 计算均值
    var_val = np.var(col)         # 计算方差
    print(f"{sub:<6}{mean_val:8.2f}{var_val:10.2f}")

# ============================================================
# 4. 步骤3：总分与 TopK 分析
# ============================================================
# 计算每位学生的总分，axis=1 表示按行求和
total_scores = np.sum(data, axis=1)

# 按总分从高到低排序的索引
order_desc = np.argsort(total_scores)[::-1]

# 计算每个学生的总排名：第1名总分最高
ranks = np.empty_like(order_desc)
ranks[order_desc] = np.arange(1, len(total_scores) + 1)

# 最高5名：总分从高到低
top5_idx = order_desc[:5]

# 最低5名：总分最低的5名，按总排名从第496到第500显示
bottom5_idx = order_desc[-5:]

# 输出总分最高的5名学生
print("\n【总分最高的5名学生】")
print(f"{'排名':<6}{'学生编号':<10}{'总分':>6}{'数学':>6}{'语文':>6}{'英语':>6}")

# 注意：下面的 for 只用于逐行打印，排序和索引提取已经由 NumPy 完成
for idx in top5_idx:
    print(f"{ranks[idx]:<6}{idx + 1:<10}{total_scores[idx]:>6}"
          f"{data[idx, 0]:>6}{data[idx, 1]:>6}{data[idx, 2]:>6}")

# 输出总分最低的5名学生
print("\n【总分最低的5名学生】")
print(f"{'排名':<6}{'学生编号':<10}{'总分':>6}{'数学':>6}{'语文':>6}{'英语':>6}")

for idx in bottom5_idx:
    print(f"{ranks[idx]:<6}{idx + 1:<10}{total_scores[idx]:>6}"
          f"{data[idx, 0]:>6}{data[idx, 1]:>6}{data[idx, 2]:>6}")