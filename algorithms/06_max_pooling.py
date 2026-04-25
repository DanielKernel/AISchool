"""
最大池化 (Max Pooling) — 详解见 STUDY_GUIDE.md。
核心实现：aischool.algorithm_core.max_pooling2d / avg_pooling2d
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from aischool.algorithm_core import avg_pooling2d, max_pooling2d


def max_pooling(image, pool_size=2, stride=2):
    return max_pooling2d(image, pool_size=pool_size, stride=stride)


def avg_pooling(image, pool_size=2, stride=2):
    return avg_pooling2d(image, pool_size=pool_size, stride=stride)


if __name__ == "__main__":
    image = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [7, 2, 3, 8],
        [4, 9, 1, 5],
    ], dtype=float)

    max_result = max_pooling(image, pool_size=2, stride=2)
    avg_result = avg_pooling(image, pool_size=2, stride=2)

    print("Max Pooling 示例:")
    print("=" * 40)
    print("输入 (4x4):")
    print(image)
    print(f"\n2x2 Max Pooling 结果 ({max_result.shape[0]}x{max_result.shape[1]}):")
    print(max_result)

    print()
    print("--- 手动验证 ---")
    print(f"  左上角 2x2: [{image[0,0]}, {image[0,1]}, {image[1,0]}, {image[1,1]}] → max = {max_result[0,0]:.0f}")
    print(f"  右上角 2x2: [{image[0,2]}, {image[0,3]}, {image[1,2]}, {image[1,3]}] → max = {max_result[0,1]:.0f}")
    print(f"  左下角 2x2: [{image[2,0]}, {image[2,1]}, {image[3,0]}, {image[3,1]}] → max = {max_result[1,0]:.0f}")
    print(f"  右下角 2x2: [{image[2,2]}, {image[2,3]}, {image[3,2]}, {image[3,3]}] → max = {max_result[1,1]:.0f}")

    print(f"\n2x2 Average Pooling 对比:")
    print(avg_result)
