"""
梯度下降法（线性回归版）
=======================

【算法背景】
梯度下降是机器学习中最基础、最重要的优化算法。几乎所有的神经网络训练都基于梯度下降。
核心思想：沿着损失函数梯度的反方向更新参数，使损失不断减小。

想象你蒙着眼站在山上，想走到最低点：
  每一步都摸一下周围哪个方向最陡（梯度），然后朝下坡方向走一步。
  步子大小（学习率）很关键：太大会走过头，太小会走太慢。

【适用场景】
- 线性回归、逻辑回归、神经网络的参数学习
- 几乎所有可微分模型的训练

【线性回归模型】
  模型:    ŷ = X @ w = w₀ + w₁x₁ + w₂x₂ + ...
  损失函数: MSE = (1/n) * Σ(ŷᵢ - yᵢ)²
  梯度:    ∂MSE/∂w = (2/n) * X^T @ (ŷ - y)

【算法流程（4 步循环）】
    ┌──────────────────────────────┐
    │ 初始化: w = 0 (或随机小值)     │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 1: ŷ = X @ w            │
    │   计算所有样本的预测值          │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 2: loss = mean((ŷ-y)²)  │◄──┐
    │   计算均方误差损失             │   │
    └──────────┬───────────────────┘   │
               ▼                       │
    ┌──────────────────────────────┐   │
    │ Step 3: g = (2/n) * X^T@(ŷ-y)│   │
    │   计算梯度                    │   │
    └──────────┬───────────────────┘   │
               ▼                       │
    ┌──────────────────────────────┐   │
    │ Step 4: w = w - lr * g        │   │
    │   沿梯度反方向更新参数         │   │
    │   收敛? → 结束                │   │
    │   未收敛 → 继续               │───┘
    └──────────────────────────────┘

【核心公式 ★★★ 必背】
  预测:   ŷ = X @ w
  误差:   error = ŷ - y
  损失:   loss = mean(error²)
  梯度:   gradient = (2/n) * X.T @ error
  更新:   w = w - lr * gradient

【口诀】算预测 → 算误差 → 算梯度 → 更新参数

【记忆要点 - 考试必背】
  1. 梯度公式: (2/n) * X.T @ (y_pred - y)
     ★ 注意是 X.T (转置)，不是 X
     ★ 分子的 2 来自对 error² 求导
     ★ 分母的 n 来自 mean 操作
  2. 更新公式: w = w - lr * gradient
     ★ 减号！梯度指向上坡方向，要朝反方向走
  3. 偏置处理: X 要加一列全 1 → w[0] 就是截距 (bias)
     X = np.column_stack([np.ones(n), x])
  4. 学习率 lr 的选择:
     太大 → 震荡发散   太小 → 收敛慢
     常用值: 0.01, 0.001

【梯度下降的三种变体（常考）】
  BGD  (批量):   用全部样本计算梯度 → 稳定但慢
  SGD  (随机):   每次用 1 个样本    → 快但不稳定
  Mini-batch:    每次用一小批样本    → 平衡（最常用）

【易错点】
  - 梯度前面有系数 2/n，不要忘记
  - X.T @ error 不是 error @ X.T（矩阵乘法顺序）
  - w 的维度 = 特征数（含偏置列则 +1）
  - 特征值范围差异大时需要先做标准化，否则梯度下降非常慢

【复杂度】
  每次迭代: O(n * d)   矩阵-向量乘法
  总体:     O(n * d * T)  T 为迭代次数
"""
import numpy as np


def gradient_descent(X, y, lr=0.01, max_iters=1000, tol=1e-6):
    """
    参数:
        X:  特征矩阵, shape (n_samples, n_features)，已含偏置列
        y:  目标值,   shape (n_samples,)
        lr: 学习率
        max_iters: 最大迭代次数
        tol: 收敛阈值
    返回:
        w: 学习到的参数, shape (n_features,)
        losses: 每次迭代的损失值列表
    """
    n = X.shape[0]
    w = np.zeros(X.shape[1])
    losses = []

    for _ in range(max_iters):
        # Step 1: 计算预测值
        y_pred = X @ w

        # Step 2: 计算损失 (MSE)
        error = y_pred - y
        loss = np.mean(error ** 2)
        losses.append(loss)

        # Step 3: 计算梯度  ★ 核心公式
        gradient = (2 / n) * (X.T @ error)

        # Step 4: 更新参数  ★ 注意是减号
        w = w - lr * gradient

        if len(losses) > 1 and abs(losses[-2] - losses[-1]) < tol:
            break

    return w, losses


# ========== 运行示例 ==========
if __name__ == "__main__":
    np.random.seed(42)

    # 生成数据: y = 3*x + 2 + 噪声
    x = np.linspace(0, 10, 50)
    y = 3 * x + 2 + np.random.randn(50) * 0.5

    # 构建特征矩阵（加偏置列: 第一列全是 1）
    X = np.column_stack([np.ones_like(x), x])

    w, losses = gradient_descent(X, y, lr=0.01, max_iters=1000)
    print("梯度下降 - 线性回归:")
    print("=" * 40)
    print(f"学习到的参数: 截距 = {w[0]:.4f}, 斜率 = {w[1]:.4f}")
    print(f"真实参数:     截距 = 2.0000, 斜率 = 3.0000")
    print(f"最终损失: {losses[-1]:.6f}")
    print(f"迭代次数: {len(losses)}")

    print()
    print("--- 理解辅助：手动计算第一次迭代 ---")
    w_init = np.zeros(2)
    y_pred = X @ w_init
    error = y_pred - y
    gradient = (2 / len(y)) * (X.T @ error)
    w_new = w_init - 0.01 * gradient
    print(f"  初始 w = {w_init}")
    print(f"  y_pred (全0)  vs  y (真实值): 误差很大")
    print(f"  梯度 = (2/n) * X.T @ error = [{gradient[0]:.4f}, {gradient[1]:.4f}]")
    print(f"  更新后 w = [{w_new[0]:.4f}, {w_new[1]:.4f}]")

    print()
    print("--- 损失下降过程 ---")
    milestones = [0, 1, 10, 50, 100, len(losses)-1]
    for m in milestones:
        if m < len(losses):
            print(f"  第 {m:>4} 次迭代: loss = {losses[m]:.6f}")
