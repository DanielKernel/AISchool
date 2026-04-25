"""
梯度下降法（线性回归版）
=======================
核心思想：预测值 = X @ w → 计算误差 → 求梯度 → 沿梯度反方向更新参数 → 重复

口诀：算预测 → 算误差 → 算梯度 → 更新参数

损失函数: MSE = (1/n) * Σ(y_pred - y)²
梯度:     ∂MSE/∂w = (2/n) * X^T @ (y_pred - y)
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
    w = np.zeros(X.shape[1])  # 初始化参数为 0
    losses = []

    for _ in range(max_iters):
        # Step 1: 计算预测值
        y_pred = X @ w

        # Step 2: 计算损失 (MSE)
        error = y_pred - y
        loss = np.mean(error ** 2)
        losses.append(loss)

        # Step 3: 计算梯度
        gradient = (2 / n) * (X.T @ error)

        # Step 4: 更新参数
        w = w - lr * gradient

        # 检查收敛
        if len(losses) > 1 and abs(losses[-2] - losses[-1]) < tol:
            break

    return w, losses


# ========== 运行示例 ==========
if __name__ == "__main__":
    np.random.seed(42)

    # 生成数据: y = 3*x + 2 + 噪声
    x = np.linspace(0, 10, 50)
    y = 3 * x + 2 + np.random.randn(50) * 0.5

    # 构建特征矩阵（加偏置列）
    X = np.column_stack([np.ones_like(x), x])  # shape: (50, 2)

    w, losses = gradient_descent(X, y, lr=0.01, max_iters=1000)
    print(f"学习到的参数: 截距 = {w[0]:.4f}, 斜率 = {w[1]:.4f}")
    print(f"真实参数:     截距 = 2.0000, 斜率 = 3.0000")
    print(f"最终损失: {losses[-1]:.6f}")
    print(f"迭代次数: {len(losses)}")
