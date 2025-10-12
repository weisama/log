import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 定义更宽的x轴范围
mu = np.linspace(25, 35, 500)

# 计算概率密度
L1 = norm.pdf(mu, 29, np.sqrt(2))  # 称1的似然，标准差为√2
L2 = norm.pdf(mu, 32, np.sqrt(4))  # 称2的似然，标准差为2
posterior = norm.pdf(mu, 30, np.sqrt(4/3))  # 后验分布，标准差为√(4/3)

# 创建图形和坐标轴
plt.figure(figsize=(12, 8))

# 绘制概率密度曲线
plt.plot(mu, L1, label='Scale 1 Likelihood (N(29, 2))', color='blue', linewidth=2)
plt.plot(mu, L2, label='Scale 2 Likelihood (N(32, 4))', color='green', linewidth=2)
plt.plot(mu, posterior, label='Posterior Distribution (N(30, 4/3))', color='red', linewidth=2)

# 标记关键点
plt.axvline(x=29, color='blue', linestyle='--', alpha=0.7, label='Measurement 1: 29g')
plt.axvline(x=32, color='green', linestyle='--', alpha=0.7, label='Measurement 2: 32g')
plt.axvline(x=30, color='red', linestyle='--', alpha=0.7, label='Posterior Mean: 30g')

# 标记真实值
plt.axvline(x=30, color='black', linestyle=':', alpha=0.5, label='True Value: 30g')

# 填充曲线下方的区域
plt.fill_between(mu, L1, alpha=0.2, color='blue')
plt.fill_between(mu, L2, alpha=0.2, color='green')
plt.fill_between(mu, posterior, alpha=0.2, color='red')

# 设置坐标轴标签和标题
plt.xlabel('True Weight μ (g)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Probability Distributions of Scale Measurements and Posterior Estimate', fontsize=14)

# 添加图例和网格
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)

# 设置坐标轴范围
plt.xlim(25, 35)
plt.ylim(0, 0.4)

# 添加文本说明
plt.text(26, 0.35, 'Scale 1: More precise (σ²=2)', color='blue', fontsize=10)
plt.text(26, 0.32, 'Scale 2: Less precise (σ²=4)', color='green', fontsize=10)
plt.text(26, 0.29, 'Posterior: Combined estimate', color='red', fontsize=10)

plt.tight_layout()
plt.show()