"""
机器学习核心算法演示（实现见 aischool.algorithm_core）。
涵盖：K-Means、KNN、梯度下降（线性回归）、SMOTE
"""

import numpy as np

from aischool.algorithm_core import (
    gradient_descent_linear_regression,
    knn_predict,
    kmeans,
    smote,
)


def demo_kmeans():
    print("=" * 50)
    print("【1. K-Means 演示】")
    np.random.seed(0)
    cluster1 = np.random.randn(30, 2) + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) + np.array([5, 5])
    cluster3 = np.random.randn(30, 2) + np.array([10, 0])
    X = np.vstack([cluster1, cluster2, cluster3])

    centers, labels = kmeans(X, K=3)
    print(f"  最终簇心:\n{centers.round(2)}")
    print(f"  各簇样本数: { {k: int((labels==k).sum()) for k in range(3)} }")


def demo_knn():
    print("=" * 50)
    print("【2. KNN 演示】")
    np.random.seed(1)
    X_train = np.vstack([
        np.random.randn(20, 2) + [0, 0],
        np.random.randn(20, 2) + [4, 4],
    ])
    y_train = np.array([0] * 20 + [1] * 20)
    X_test = np.array([[0.5, 0.5], [3.5, 3.5], [2, 2]])

    preds = knn_predict(X_train, y_train, X_test, K=3)
    print(f"  测试样本: {X_test.tolist()}")
    print(f"  KNN(K=3) 预测结果: {preds.tolist()}")
    print(f"  预期结果: [0, 1, 0或1（边界点）]")


def demo_gradient_descent():
    print("=" * 50)
    print("【3. 梯度下降（线性回归）演示】")
    np.random.seed(2)
    X = np.random.randn(100, 1)
    y = 2 * X.ravel() + 1 + np.random.randn(100) * 0.5

    w, b, losses = gradient_descent_linear_regression(X, y, lr=0.1, epochs=1000)
    print(f"  学习到的参数: w={w[0]:.4f}, b={b:.4f}")
    print(f"  真实参数:     w=2.0,    b=1.0")
    print(f"  最终 MSE 损失: {losses[-1]:.4f}")


def demo_smote():
    print("=" * 50)
    print("【4. SMOTE 演示】")
    np.random.seed(3)
    X_minority = np.random.randn(10, 2) + [8, 8]
    X_majority = np.random.randn(50, 2) + [0, 0]

    print(f"  过采样前 - 少数类: {len(X_minority)}, 多数类: {len(X_majority)}")

    synthetic = smote(X_minority, n_synthetic=40, K=5)
    X_minority_augmented = np.vstack([X_minority, synthetic])

    print(f"  过采样后 - 少数类: {len(X_minority_augmented)}, 多数类: {len(X_majority)}")
    print(f"  合成样本中心（应接近 [8,8]）: {synthetic.mean(axis=0).round(2)}")


if __name__ == "__main__":
    print("\n==================================================")
    print("   机器学习算法演示（K-Means / KNN / 梯度下降 / SMOTE）")
    print("==================================================\n")

    demo_kmeans()
    print()
    demo_knn()
    print()
    demo_gradient_descent()
    print()
    demo_smote()
    print()
    print("所有机器学习算法演示完毕！")
