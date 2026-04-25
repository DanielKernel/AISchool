"""
2D 卷积（滑动窗口）— 详解见 STUDY_GUIDE.md。
核心实现：aischool.algorithm_core.conv2d
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from aischool.algorithm_core import conv2d as _conv2d


def conv2d(image, kernel, stride=1, padding=0):
    return _conv2d(image, kernel, stride=stride, padding=padding)


if __name__ == "__main__":
    image = np.array([
        [1, 2, 3, 0, 1],
        [0, 1, 2, 3, 1],
        [1, 3, 1, 0, 2],
        [2, 1, 0, 1, 3],
        [0, 2, 1, 2, 1],
    ], dtype=float)

    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ], dtype=float)

    result = conv2d(image, kernel)
    print("2D 卷积示例:")
    print("=" * 40)
    print("输入图像 (5x5):")
    print(image)
    print(f"\n卷积核 (3x3) - 边缘检测:")
    print(kernel)
    print(f"\n输出尺寸: ({image.shape[0]}-3)//1 + 1 = {result.shape[0]}")
    print(f"卷积结果 ({result.shape[0]}x{result.shape[1]}):")
    print(result)

    print()
    print("--- 理解辅助：手动计算 output[0,0] ---")
    region = image[0:3, 0:3]
    print(f"  窗口 region = image[0:3, 0:3]:")
    print(f"    {region[0]}")
    print(f"    {region[1]}")
    print(f"    {region[2]}")
    print(f"  region * kernel (逐元素乘):")
    product = region * kernel
    print(f"    {product[0]}")
    print(f"    {product[1]}")
    print(f"    {product[2]}")
    print(f"  sum = {np.sum(product):.0f}")

    print()
    print("--- padding 示例 ---")
    result_pad = conv2d(image, kernel, padding=1)
    print(f"  padding=1 时输出尺寸: {result_pad.shape[0]}x{result_pad.shape[1]} (与输入相同!)")
