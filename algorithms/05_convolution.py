"""
2D 卷积（滑动窗口）
==================
核心思想：卷积核在输入矩阵上滑动 → 对应位置逐元素相乘 → 求和得到输出的一个值

口诀：双层循环 → 提取窗口 → 逐元素乘 → 求和
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
    # Step 1: 填充
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)

    H, W = image.shape
    kH, kW = kernel.shape

    # Step 2: 计算输出尺寸
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    output = np.zeros((out_H, out_W))

    # Step 3: 滑动窗口
    for i in range(out_H):
        for j in range(out_W):
            # 提取当前窗口区域
            region = image[i*stride : i*stride+kH, j*stride : j*stride+kW]
            # 逐元素相乘后求和
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
    print("输入图像 (5x5):")
    print(image)
    print(f"\n卷积核 (3x3):")
    print(kernel)
    print(f"\n卷积结果 ({result.shape[0]}x{result.shape[1]}):")
    print(result)
