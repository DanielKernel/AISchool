"""
K 近邻 (KNN) 分类算法
====================
核心思想：计算待分类点与所有训练样本的欧氏距离 → 取最近的 K 个 → 多数投票决定类别

口诀：算距离 → 排序 → 投票
"""
import numpy as np
from collections import Counter


def knn_predict(X_train, y_train, x_query, k=3):
    """
    对单个查询点进行 KNN 分类预测

    参数:
        X_train: 训练数据, shape (n_samples, n_features)
        y_train: 训练标签, shape (n_samples,)
        x_query: 待分类的单个样本, shape (n_features,)
        k: 近邻数量
    返回:
        预测的类别标签
    """
    # Step 1: 计算查询点到所有训练样本的欧氏距离
    distances = np.linalg.norm(X_train - x_query, axis=1)

    # Step 2: 找到距离最小的 k 个样本的索引
    k_nearest_indices = np.argsort(distances)[:k]

    # Step 3: 取这 k 个样本的标签，多数投票
    k_nearest_labels = y_train[k_nearest_indices]
    most_common = Counter(k_nearest_labels).most_common(1)

    return most_common[0][0]


def knn_predict_batch(X_train, y_train, X_test, k=3):
    """对一批测试样本进行预测"""
    return np.array([knn_predict(X_train, y_train, x, k) for x in X_test])


# ========== 运行示例 ==========
if __name__ == "__main__":
    np.random.seed(42)
    # 训练数据: 两类点
    X_train = np.array([[1, 1], [1, 2], [2, 1],   # 类别 0
                         [5, 5], [5, 6], [6, 5]])   # 类别 1
    y_train = np.array([0, 0, 0, 1, 1, 1])

    # 测试
    test_points = np.array([[1.5, 1.5], [5.5, 5.5], [3, 3]])
    for pt in test_points:
        pred = knn_predict(X_train, y_train, pt, k=3)
        print(f"点 ({pt[0]}, {pt[1]}) → 预测类别: {pred}")
