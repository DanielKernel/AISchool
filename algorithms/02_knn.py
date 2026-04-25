"""
K 近邻 (KNN) — 详解见 STUDY_GUIDE.md。
核心实现：aischool.algorithm_core.knn_predict_one / knn_predict_batch
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from collections import Counter

from aischool.algorithm_core import knn_predict_batch as _knn_batch
from aischool.algorithm_core import knn_predict_one


def knn_predict(X_train, y_train, x_query, k=3):
    return knn_predict_one(X_train, y_train, x_query, k=k)


def knn_predict_batch(X_train, y_train, X_test, k=3):
    return _knn_batch(X_train, y_train, X_test, k=k)


if __name__ == "__main__":
    np.random.seed(42)
    X_train = np.array([[1, 1], [1, 2], [2, 1],
                        [5, 5], [5, 6], [6, 5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    print("KNN 分类结果:")
    print("=" * 50)
    test_points = np.array([[1.5, 1.5], [5.5, 5.5], [3, 3]])
    for pt in test_points:
        pred = knn_predict(X_train, y_train, pt, k=3)
        print(f"  点 ({pt[0]}, {pt[1]}) → 预测类别: {pred}")

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
