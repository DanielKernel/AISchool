"""
机器学习核心算法实现
涵盖：K-Means、KNN、梯度下降（线性回归）、SMOTE
每个算法都仅用标准库 + numpy，不依赖 sklearn，方便考试手写理解。
"""

import numpy as np
import random


# ============================================================
# 1. K-Means（簇心迭代聚类）
# ============================================================
# 原理：
#   随机选 K 个中心点 → 每个样本分配给最近中心 → 更新中心为簇均值
#   重复直到中心不再变化（或达到最大迭代次数）
# 关键：欧氏距离 + 均值更新
# ============================================================

def euclidean_distance(a, b):
    """欧氏距离（L2）"""
    return np.sqrt(np.sum((a - b) ** 2))


def kmeans(X, K, max_iter=100, random_state=42):
    """
    K-Means 聚类

    参数：
        X           : ndarray, shape (n_samples, n_features)
        K           : 簇的数量
        max_iter    : 最大迭代次数
        random_state: 随机种子

    返回：
        centers : 最终簇心，shape (K, n_features)
        labels  : 每个样本所属簇的索引，shape (n_samples,)
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]

    # Step 1：从数据中随机选 K 个样本作为初始中心
    idx = np.random.choice(n_samples, K, replace=False)
    centers = X[idx].copy()

    labels = np.zeros(n_samples, dtype=int)

    for iteration in range(max_iter):
        # Step 2：分配——每个点分给距离最近的中心
        new_labels = np.array([
            np.argmin([euclidean_distance(x, c) for c in centers])
            for x in X
        ])

        # Step 3：更新——每个簇的新中心 = 该簇所有点的均值
        new_centers = np.array([
            X[new_labels == k].mean(axis=0) if np.any(new_labels == k) else centers[k]
            for k in range(K)
        ])

        # Step 4：收敛检测——中心不再变化则停止
        if np.allclose(centers, new_centers):
            print(f"  K-Means 第 {iteration+1} 次迭代后收敛")
            break

        centers = new_centers
        labels = new_labels

    return centers, new_labels


def demo_kmeans():
    print("=" * 50)
    print("【1. K-Means 演示】")
    # 构造三个明显分离的簇
    np.random.seed(0)
    cluster1 = np.random.randn(30, 2) + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) + np.array([5, 5])
    cluster3 = np.random.randn(30, 2) + np.array([10, 0])
    X = np.vstack([cluster1, cluster2, cluster3])

    centers, labels = kmeans(X, K=3)
    print(f"  最终簇心:\n{centers.round(2)}")
    print(f"  各簇样本数: { {k: int((labels==k).sum()) for k in range(3)} }")


# ============================================================
# 2. KNN（K 近邻分类）
# ============================================================
# 原理：
#   预测新样本时，找训练集中欧氏距离最近的 K 个邻居
#   取邻居中出现次数最多的类别作为预测结果（投票）
# 关键：无训练过程（lazy learning），预测时计算所有距离
# ============================================================

def knn_predict(X_train, y_train, X_test, K=3):
    """
    KNN 分类预测

    参数：
        X_train : 训练特征，shape (n_train, n_features)
        y_train : 训练标签，shape (n_train,)
        X_test  : 测试特征，shape (n_test, n_features)
        K       : 邻居数

    返回：
        predictions : 预测标签列表
    """
    predictions = []
    for x in X_test:
        # 计算 x 与所有训练点的距离
        distances = [euclidean_distance(x, xi) for xi in X_train]

        # 取距离最小的 K 个邻居的索引
        k_indices = np.argsort(distances)[:K]
        k_labels = y_train[k_indices]

        # 投票：取出现次数最多的标签
        values, counts = np.unique(k_labels, return_counts=True)
        predicted_label = values[np.argmax(counts)]
        predictions.append(predicted_label)

    return np.array(predictions)


def demo_knn():
    print("=" * 50)
    print("【2. KNN 演示】")
    # 构造二分类数据
    np.random.seed(1)
    X_train = np.vstack([
        np.random.randn(20, 2) + [0, 0],   # 类别 0
        np.random.randn(20, 2) + [4, 4],   # 类别 1
    ])
    y_train = np.array([0] * 20 + [1] * 20)

    # 测试样本
    X_test = np.array([[0.5, 0.5], [3.5, 3.5], [2, 2]])

    preds = knn_predict(X_train, y_train, X_test, K=3)
    print(f"  测试样本: {X_test.tolist()}")
    print(f"  KNN(K=3) 预测结果: {preds.tolist()}")
    print(f"  预期结果: [0, 1, 0或1（边界点）]")


# ============================================================
# 3. 梯度下降（线性回归版）
# ============================================================
# 原理：
#   损失函数 MSE = mean((Xw + b - y)^2)
#   对 w 和 b 分别求偏导（梯度）
#   沿梯度反方向更新参数：w = w - lr * grad_w
# 关键：学习率 lr 控制步长
# ============================================================

def gradient_descent_linear_regression(X, y, lr=0.01, epochs=1000):
    """
    梯度下降求解线性回归

    参数：
        X      : 特征矩阵，shape (n_samples, n_features)
        y      : 目标值，shape (n_samples,)
        lr     : 学习率
        epochs : 迭代轮数

    返回：
        w : 权重向量，shape (n_features,)
        b : 偏置标量
        loss_history : 每轮的 MSE 损失
    """
    n, d = X.shape
    w = np.zeros(d)   # 初始化权重为 0
    b = 0.0           # 初始化偏置为 0
    loss_history = []

    for epoch in range(epochs):
        # 前向传播：预测值
        y_pred = X @ w + b

        # 计算损失（MSE）
        error = y_pred - y
        loss = np.mean(error ** 2)
        loss_history.append(loss)

        # 计算梯度
        grad_w = (X.T @ error) / n   # ∂L/∂w
        grad_b = error.mean()         # ∂L/∂b

        # 参数更新（梯度下降）
        w -= lr * grad_w
        b -= lr * grad_b

        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1:4d} | MSE Loss = {loss:.4f}")

    return w, b, loss_history


def demo_gradient_descent():
    print("=" * 50)
    print("【3. 梯度下降（线性回归）演示】")
    np.random.seed(2)
    # 真实关系：y = 2*x + 1 + 噪声
    X = np.random.randn(100, 1)
    y = 2 * X.ravel() + 1 + np.random.randn(100) * 0.5

    w, b, losses = gradient_descent_linear_regression(X, y, lr=0.1, epochs=1000)
    print(f"  学习到的参数: w={w[0]:.4f}, b={b:.4f}")
    print(f"  真实参数:     w=2.0,    b=1.0")
    print(f"  最终 MSE 损失: {losses[-1]:.4f}")


# ============================================================
# 4. SMOTE（合成少数过采样技术）
# ============================================================
# 原理：
#   对少数类每个样本，在其 K 近邻中随机选一个邻居
#   在该样本与邻居之间的连线上随机插值，生成新合成样本
# 关键公式：new = x + λ * (xn - x)，λ ~ Uniform(0, 1)
# ============================================================

def smote(X_minority, n_synthetic, K=5, random_state=42):
    """
    SMOTE 过采样

    参数：
        X_minority  : 少数类样本，shape (n_minority, n_features)
        n_synthetic : 需要生成的合成样本数量
        K           : 近邻数

    返回：
        synthetic_samples : 合成样本，shape (n_synthetic, n_features)
    """
    np.random.seed(random_state)
    n_samples = X_minority.shape[0]
    synthetic_samples = []

    for _ in range(n_synthetic):
        # 随机选一个少数类样本作为基准点
        idx = np.random.randint(0, n_samples)
        x = X_minority[idx]

        # 找该样本的 K 个最近邻（排除自身）
        distances = [euclidean_distance(x, xi) for xi in X_minority]
        distances[idx] = np.inf   # 排除自身
        k_indices = np.argsort(distances)[:K]

        # 随机选一个邻居
        neighbor_idx = np.random.choice(k_indices)
        xn = X_minority[neighbor_idx]

        # 在 x 和 xn 之间随机插值
        lam = np.random.uniform(0, 1)
        synthetic = x + lam * (xn - x)
        synthetic_samples.append(synthetic)

    return np.array(synthetic_samples)


def demo_smote():
    print("=" * 50)
    print("【4. SMOTE 演示】")
    np.random.seed(3)
    # 少数类：只有 10 个样本，分布在 (8, 8) 附近
    X_minority = np.random.randn(10, 2) + [8, 8]
    # 多数类：50 个样本
    X_majority = np.random.randn(50, 2) + [0, 0]

    print(f"  过采样前 - 少数类: {len(X_minority)}, 多数类: {len(X_majority)}")

    # 生成 40 个合成少数类样本，使两类平衡
    synthetic = smote(X_minority, n_synthetic=40, K=5)
    X_minority_augmented = np.vstack([X_minority, synthetic])

    print(f"  过采样后 - 少数类: {len(X_minority_augmented)}, 多数类: {len(X_majority)}")
    print(f"  合成样本中心（应接近 [8,8]）: {synthetic.mean(axis=0).round(2)}")


# ============================================================
# 主程序
# ============================================================

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
