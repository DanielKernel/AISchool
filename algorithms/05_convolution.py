"""
2D 卷积（滑动窗口）
==================

【算法背景】
卷积是卷积神经网络 (CNN) 的核心操作。在图像处理中，卷积用于提取图像的局部特征
（如边缘、纹理、角点等）。不同的卷积核可以检测不同类型的特征。

数学上，2D 卷积是两个矩阵的一种运算：
  用一个小矩阵（卷积核/滤波器）在大矩阵（输入图像）上滑动，
  在每个位置计算对应元素的乘积之和。

【适用场景】
- 图像特征提取、边缘检测、模糊/锐化滤波
- CNN 中的特征学习层

【算法流程】
    ┌──────────────────────────────┐
    │ Step 0: (可选) 对输入做零填充   │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 1: 计算输出尺寸           │
    │  out = (H - kH + 2*pad)/s + 1 │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 2: 双层循环滑动窗口       │
    │  for i in range(out_H):      │
    │    for j in range(out_W):    │
    │      提取 region             │
    │      output[i,j] =          │
    │        Σ(region * kernel)    │
    └──────────────────────────────┘

【核心公式】
  输出尺寸: out = (input_size - kernel_size + 2*padding) / stride + 1
  卷积运算: output[i,j] = Σₘ Σₙ image[i*s+m, j*s+n] * kernel[m, n]

  ★ 公式记忆法: "输入减核加两倍填充，除步长加一"

【口诀】双层循环 → 提取窗口 → 逐元素乘 → 求和

【记忆要点 - 考试必背】
  1. 输出尺寸公式: out = (H - kH + 2*pad) // stride + 1
     ★ 这个公式考试经常直接考！
  2. 零填充: np.pad(image, padding, mode='constant', constant_values=0)
  3. 窗口提取: region = image[i*s : i*s+kH, j*s : j*s+kW]
     ★ 起始位置 = i * stride，长度 = kernel_size
  4. 卷积计算: np.sum(region * kernel)
     ★ 逐元素相乘（不是矩阵乘法！）然后求和

【常见卷积核（了解）】
  边缘检测:    [[-1,-1,-1],     均值模糊:  [[1,1,1],
                [-1, 8,-1],                 [1,1,1],
                [-1,-1,-1]]                 [1,1,1]] / 9

  锐化:        [[ 0,-1, 0],     水平边缘:  [[-1,-1,-1],
                [-1, 5,-1],                 [ 0, 0, 0],
                [ 0,-1, 0]]                 [ 1, 1, 1]]

【易错点】
  - 卷积 vs 相关：严格的卷积需要把核翻转 180°，但 CNN 中通常做的是"互相关"（不翻转）
  - 输出尺寸：整除用 // 不是 /
  - padding=0 时输出会缩小！padding='same' 时输出和输入相同大小
  - region * kernel 是逐元素乘 (Hadamard 积)，不是矩阵乘法 (@)

【与池化的关系】
  卷积和池化共享相同的滑动窗口框架，区别仅在窗口内的操作:
  - 卷积: Σ(region * kernel) → 加权求和
  - 池化: max(region)        → 取最大值

【复杂度】
  时间: O(out_H * out_W * kH * kW)   每个输出位置做 kH*kW 次乘法
  空间: O(out_H * out_W)             输出矩阵
"""
import numpy as np


def conv2d(image, kernel, stride=1, padding=0):
    """
    参数:
        image:   输入矩阵, shape (H, W)
        kernel:  卷积核,   shape (kH, kW)
        stride:  步长
        padding: 零填充大小
    返回:
        output:  卷积结果, shape (out_H, out_W)
    """
    # Step 0: 填充
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)

    H, W = image.shape
    kH, kW = kernel.shape

    # Step 1: 计算输出尺寸  ★ 必背公式
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    output = np.zeros((out_H, out_W))

    # Step 2: 滑动窗口
    for i in range(out_H):
        for j in range(out_W):
            region = image[i*stride : i*stride+kH, j*stride : j*stride+kW]
            output[i, j] = np.sum(region * kernel)

    return output


# ========== 运行示例 ==========
if __name__ == "__main__":
    # 5x5 的输入图像
    image = np.array([
        [1, 2, 3, 0, 1],
        [0, 1, 2, 3, 1],
        [1, 3, 1, 0, 2],
        [2, 1, 0, 1, 3],
        [0, 2, 1, 2, 1],
    ], dtype=float)

    # 3x3 边缘检测核
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
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
