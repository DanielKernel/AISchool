"""
常见 AI/ML 算法的最小可运行 Python 核心实现，用于考试复习与自检。
实现位于 aischool.algorithm_core（与 algorithms/、根目录演示共用，避免重复）。

运行: python examples/algorithms_reference.py
依赖: numpy
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from aischool.algorithm_core import (
    SimpleDecisionStump,
    conv2d_valid,
    dtw_distance,
    dtw_warping_path,
    is_bipartite_edges,
    knn_classify,
    kmeans_vectorized,
    kruskal_mst,
    linear_regression_gd,
    linear_regression_gd_recursive,
    max_pool2d,
    pagerank_power_iteration,
    smote_reference,
)


def _self_check() -> None:
    rng = np.random.default_rng(42)

    c1 = rng.normal(0, 0.3, (30, 2))
    c2 = rng.normal(3, 0.3, (30, 2))
    Xk = np.vstack([c1, c2])
    centers, labels = kmeans_vectorized(Xk, k=2, seed=1)
    assert centers.shape == (2, 2) and labels.shape == (60,)

    yk = np.array([0] * 30 + [1] * 30)
    pred = knn_classify(Xk, yk, np.array([0.0, 0.0]), k=5)
    assert pred == 0

    pr = pagerank_power_iteration({0: [1], 1: [2], 2: [0]}, 3)
    assert abs(pr.sum() - 1.0) < 1e-6

    Xd = np.array([[1], [2], [3], [10]])
    yd = np.array([0, 0, 1, 1])
    stump = SimpleDecisionStump()
    stump.fit(Xd, yd)
    assert stump.predict_row(np.array([1.5])) in (0, 1)

    img = np.arange(16, dtype=float).reshape(4, 4)
    ker = np.ones((2, 2))
    c = conv2d_valid(img, ker)
    assert c.shape == (3, 3)
    p = max_pool2d(img, 2, 2, stride=2)
    assert p.shape == (2, 2)

    assert is_bipartite_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)]) is True
    assert is_bipartite_edges(3, [(0, 1), (1, 2), (2, 0)]) is False

    mst = kruskal_mst(4, [(0, 1, 1), (1, 2, 2), (0, 2, 3), (2, 3, 1)])
    assert len(mst) == 3

    Xm = np.column_stack([np.linspace(0, 1, 20), np.ones(20)])
    ym = 3 * Xm[:, 0] + 2 + rng.normal(0, 0.05, 20)
    w = linear_regression_gd(Xm, ym, lr=0.5, epochs=2000)
    assert abs(w[0] - 3.0) < 0.2

    w_rec = linear_regression_gd_recursive(Xm, ym, lr=0.5, epochs=200)
    assert np.allclose(w_rec, linear_regression_gd(Xm, ym, lr=0.5, epochs=200))

    s = np.array([1.0, 2.0, 3.0])
    assert dtw_distance(s, s, squared=True) == 0.0
    cst, pth = dtw_warping_path(s, s, squared=True)
    assert cst == 0.0 and pth == [(0, 0), (1, 1), (2, 2)]

    Xmin = rng.normal(size=(8, 3))
    syn = smote_reference(Xmin, k=3, n_synthetic=5, seed=0)
    assert syn.shape == (5, 3)

    print("algorithms_reference: 所有自检通过。")


if __name__ == "__main__":
    _self_check()
