"""
K 近邻 (KNN) 分类算法
====================

【算法背景】
KNN 是最简单直观的监督学习分类算法，由 Thomas Cover 于 1967 年提出。
核心思想："物以类聚" — 一个样本的类别由它周围最近的 K 个邻居投票决定。
KNN 是一种"懒惰学习"(Lazy Learning)方法：没有显式的训练过程，预测时才做计算。

【适用场景】
- 小规模数据集的分类、推荐系统、手写数字识别（MNIST）

【算法流程（3 步）】
    ┌──────────────────────────────┐
    │ Step 1: 计算待分类点到所有训练   │
    │   样本的欧氏距离                │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 2: 按距离从小到大排序，    │
    │   取前 K 个最近邻              │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 3: 这 K 个邻居的标签      │
    │   多数投票 → 得到预测类别       │
    └──────────────────────────────┘

【核心公式】
  欧氏距离:  d(x, y) = ||x - y||₂ = sqrt(Σ(xᵢ - yᵢ)²)
  投票决策:  ŷ = mode({y₁, y₂, ..., yₖ})  (出现次数最多的标签)

【口诀】算距离 → 排序 → 投票

【记忆要点 - 考试必背】
  1. 距离计算: np.linalg.norm(X_train - x_query, axis=1)
     - X_train (n, d) 减去 x_query (d,) → 广播后逐行求范数
     - axis=1 表示对每一行（每个训练样本）计算距离
  2. 排序取 K 个: np.argsort(distances)[:k]
     - argsort 返回排序后的索引，取前 k 个就是最近的 k 个邻居
  3. 投票: Counter(labels).most_common(1)[0][0]
     - Counter 统计每个标签出现次数
     - most_common(1) 返回出现最多的 [(标签, 次数)]
     - [0][0] 取标签值

【K 值选择】
  - K 太小 → 容易受噪声影响（过拟合）
  - K 太大 → 决策边界过于平滑（欠拟合）
  - 经验法则: K = sqrt(n)，且 K 取奇数避免平票

【易错点】
  - KNN 不需要训练！"训练"就是存储数据
  - 距离计算时 axis 参数方向不能搞错
  - 特征量纲不同时需要先做标准化 (StandardScaler)
  - K=1 时退化为最近邻分类器

【复杂度】
  训练: O(1)        只存储数据，不做计算
  预测: O(n * d)    计算到所有 n 个训练样本的 d 维距离
  空间: O(n * d)    存储全部训练数据

【与 K-Means 的区别（常考）】
  KNN:     有监督，分类算法，K 是近邻数
  K-Means: 无监督，聚类算法，K 是簇数
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

    print("KNN 分类结果:")
    print("=" * 50)
    test_points = np.array([[1.5, 1.5], [5.5, 5.5], [3, 3]])
    for pt in test_points:
        pred = knn_predict(X_train, y_train, pt, k=3)
        print(f"  点 ({pt[0]}, {pt[1]}) → 预测类别: {pred}")

    # 手动追踪一次预测过程，帮助理解
    print()
    print("--- 理解辅助：对点 (3, 3) 的详细预测过程 ---")
    query = np.array([3, 3])
    distances = np.linalg.norm(X_train - query, axis=1)
    for i, (pt, d, label) in enumerate(zip(X_train, distances, y_train)):
        print(f"  训练样本 {i}: ({pt[0]}, {pt[1]}), 距离={d:.2f}, 标签={label}")
    k = 3
    k_idx = np.argsort(distances)[:k]
    print(f"  最近 {k} 个邻居索引: {k_idx}, 标签: {y_train[k_idx]}")
    print(f"  投票结果: {Counter(y_train[k_idx]).most_common()}")
