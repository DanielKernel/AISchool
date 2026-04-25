"""
最大池化 (Max Pooling)
=====================
核心思想：滑动窗口在输入矩阵上移动 → 取窗口内的最大值作为输出

口诀：双层循环 → 提取窗口 → 取最大值
"""
import numpy as np


def max_pooling(image, pool_size=2, stride=2):
    """
    参数:
        image:     输入矩阵, shape (H, W)
        pool_size: 池化窗口大小
        stride:    步长 (通常等于 pool_size)
    返回:
        output:    池化结果
    """
    H, W = image.shape

    # 计算输出尺寸
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = np.zeros((out_H, out_W))

    # 滑动窗口取最大值
    for i in range(out_H):
        for j in range(out_W):
            region = image[i*stride : i*stride+pool_size,
                           j*stride : j*stride+pool_size]
            output[i, j] = np.max(region)

    return output


# ========== 运行示例 ==========
if __name__ == "__main__":
    image = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [7, 2, 3, 8],
        [4, 9, 1, 5],
    ], dtype=float)

    result = max_pooling(image, pool_size=2, stride=2)
    print("输入 (4x4):")
    print(image)
    print(f"\n2x2 Max Pooling 结果 ({result.shape[0]}x{result.shape[1]}):")
    print(result)
    # 期望结果: [[6, 4], [9, 8]]
