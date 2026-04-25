"""
决策树与深度学习基础演示（实现见 aischool.algorithm_core）。
涵盖：决策树核心（信息增益）、卷积（滑动窗口）、Max Pooling（最大池化）
"""

import numpy as np

from aischool.algorithm_core import (
    avg_pooling2d,
    best_split_feature,
    conv2d,
    entropy,
    information_gain,
    max_pooling2d,
)


def demo_decision_tree():
    print("=" * 50)
    print("【1. 决策树信息增益演示】")
    X = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],
        [2, 0], [2, 1], [0, 0], [2, 0],
        [1, 0], [0, 1],
    ])
    y = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])

    print(f"  数据集大小: {len(y)} 条样本")
    print(f"  父节点熵 H(S) = {entropy(y):.4f}")

    gain_weather = information_gain(X[:, 0], y)
    gain_wind = information_gain(X[:, 1], y)
    print(f"  特征 [天气] 信息增益 = {gain_weather:.4f}")
    print(f"  特征 [风力] 信息增益 = {gain_wind:.4f}")

    best_idx, best_gain = best_split_feature(X, y)
    feat_names = ["天气", "风力"]
    print(f"  最优分裂特征: [{feat_names[best_idx]}]，增益 = {best_gain:.4f}")


def demo_conv2d():
    print("=" * 50)
    print("【2. 卷积（滑动窗口）演示】")
    input_map = np.array([
        [1, 2, 3, 0],
        [0, 1, 2, 3],
        [3, 0, 1, 2],
        [2, 3, 0, 1],
    ], dtype=float)

    kernel = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ], dtype=float)

    output = conv2d(input_map, kernel, stride=1, padding=0)
    print(f"  输入形状: {input_map.shape}")
    print(f"  卷积核形状: {kernel.shape}，步长=1，padding=0")
    print(f"  输出形状: {output.shape}  （公式：(4-3)/1+1 = 2）")
    print(f"  输出:\n{output}")

    output_padded = conv2d(input_map, kernel, stride=1, padding=1)
    print(f"\n  padding=1 后输出形状: {output_padded.shape}  （公式：(4-3+2)/1+1 = 4）")


def demo_pooling():
    print("=" * 50)
    print("【3. Max Pooling 演示】")
    input_map = np.array([
        [1, 3, 2, 4],
        [5, 6, 7, 8],
        [3, 2, 1, 0],
        [1, 2, 3, 4],
    ], dtype=float)

    print(f"  输入 (4×4):\n{input_map}")

    mp_out = max_pooling2d(input_map, pool_size=2, stride=2)
    print(f"\n  Max Pooling（窗口=2, 步长=2）输出形状: {mp_out.shape}")
    print(f"  输出:\n{mp_out}")

    ap_out = avg_pooling2d(input_map, pool_size=2, stride=2)
    print(f"\n  Average Pooling 对比输出:\n{ap_out}")

    mp_stride1 = max_pooling2d(input_map, pool_size=2, stride=1)
    print(f"\n  Max Pooling（步长=1，重叠）输出形状: {mp_stride1.shape}")
    print(f"  输出:\n{mp_stride1}")


if __name__ == "__main__":
    print("\n==================================================")
    print("   深度学习基础演示（决策树信息增益 / 卷积 / Max Pooling）")
    print("==================================================\n")

    demo_decision_tree()
    print()
    demo_conv2d()
    print()
    demo_pooling()
    print()
    print("所有深度学习基础算法演示完毕！")
