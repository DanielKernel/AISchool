"""
批量运行所有算法示例
====================

使用方法:
    python3 run_all.py          # 运行全部算法
    python3 run_all.py 01 03    # 只运行指定编号的算法
    python3 run_all.py --list   # 列出所有可用算法
"""
import subprocess
import sys
import os

ALGORITHMS = {
    "01": ("K-Means 聚类", "algorithms/01_kmeans.py"),
    "02": ("KNN 分类", "algorithms/02_knn.py"),
    "03": ("PageRank", "algorithms/03_pagerank.py"),
    "04": ("决策树 (ID3)", "algorithms/04_decision_tree.py"),
    "05": ("2D 卷积", "algorithms/05_convolution.py"),
    "06": ("Max Pooling", "algorithms/06_max_pooling.py"),
    "07": ("二分图判断", "algorithms/07_bipartite_check.py"),
    "08": ("Kruskal MST", "algorithms/08_kruskal.py"),
    "09": ("梯度下降", "algorithms/09_gradient_descent.py"),
    "10": ("SMOTE 过采样", "algorithms/10_smote.py"),
}


def list_algorithms():
    print("可用算法:")
    print("-" * 45)
    for num, (name, path) in ALGORITHMS.items():
        print(f"  {num}  {name:<20s} {path}")


def run_algorithm(num, name, path):
    separator = "=" * 55
    print(f"\n{separator}")
    print(f"  [{num}] {name}")
    print(f"{separator}")
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        print(f"[错误] 退出码 {result.returncode}")
        if result.stderr:
            print(result.stderr, end="")
        return False
    return True


def main():
    args = sys.argv[1:]

    if "--list" in args or "-l" in args:
        list_algorithms()
        return

    if args:
        selected = {k: v for k, v in ALGORITHMS.items() if k in args}
        if not selected:
            print(f"错误: 未找到编号 {args}，使用 --list 查看可用编号")
            return
    else:
        selected = ALGORITHMS

    passed = 0
    failed = []

    for num, (name, path) in selected.items():
        ok = run_algorithm(num, name, path)
        if ok:
            passed += 1
        else:
            failed.append(f"{num} {name}")

    total = len(selected)
    print(f"\n{'=' * 55}")
    print(f"运行结果: {passed}/{total} 通过")
    if failed:
        print("失败:")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
