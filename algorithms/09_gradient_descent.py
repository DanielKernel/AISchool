"""
梯度下降法（线性回归版）— 详解见 STUDY_GUIDE.md。
核心实现：aischool.algorithm_core.gradient_descent
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from aischool.algorithm_core import gradient_descent as _gd


def gradient_descent(X, y, lr=0.01, max_iters=1000, tol=1e-6):
    return _gd(X, y, lr=lr, max_iters=max_iters, tol=tol)


if __name__ == "__main__":
    np.random.seed(42)

    x = np.linspace(0, 10, 50)
    y = 3 * x + 2 + np.random.randn(50) * 0.5
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
    milestones = [0, 1, 10, 50, 100, len(losses) - 1]
    for m in milestones:
        if m < len(losses):
            print(f"  第 {m:>4} 次迭代: loss = {losses[m]:.6f}")
