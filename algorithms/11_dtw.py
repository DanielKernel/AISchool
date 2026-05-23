"""
动态时间规整 (DTW) — 详解见 STUDY_GUIDE.md。
核心实现：aischool.algorithm_core.dtw_distance / dtw_warping_path
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from aischool.algorithm_core import dtw_distance as _dtw_distance
from aischool.algorithm_core import dtw_warping_path


def dtw(x, y, squared=True):
    """与 `dtw_distance` 同义，便于 `from algorithms import dtw` 式调用。"""
    return _dtw_distance(x, y, squared=squared)


if __name__ == "__main__":
    a = np.array([1.0, 2.0, 3.0, 3.0, 4.0])
    b = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 4.0])
    d = _dtw_distance(a, b, squared=True)
    cost, path = dtw_warping_path(a, b, squared=True)
    print("DTW 示例:")
    print("=" * 40)
    print(f"  序列 a: {a}")
    print(f"  序列 b: {b}")
    print(f"  累计平方代价 dtw_distance = {d:.4f}")
    print(f"  dtw_warping_path 代价一致: {abs(cost - d) < 1e-9}")
    print(f"  对齐路径长度: {len(path)}")
    print(f"  路径前 5 步: {path[:5]}")
