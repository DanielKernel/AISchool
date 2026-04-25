"""
一键运行所有 AI 算法演示
执行方式：python run_all.py
"""

import sys
import traceback


def run_module(module_name, description):
    """运行一个模块的演示，并捕获异常"""
    print(f"\n{'#' * 60}")
    print(f"#  {description}")
    print(f"{'#' * 60}")
    try:
        if module_name == "machine_learning":
            from machine_learning import (
                demo_kmeans, demo_knn,
                demo_gradient_descent, demo_smote
            )
            demo_kmeans()
            print()
            demo_knn()
            print()
            demo_gradient_descent()
            print()
            demo_smote()

        elif module_name == "graph_algorithms":
            from graph_algorithms import (
                demo_pagerank, demo_bipartite, demo_kruskal
            )
            demo_pagerank()
            print()
            demo_bipartite()
            print()
            demo_kruskal()

        elif module_name == "dl_basics":
            from dl_basics import (
                demo_decision_tree, demo_conv2d, demo_pooling
            )
            demo_decision_tree()
            print()
            demo_conv2d()
            print()
            demo_pooling()

        print(f"\n[OK] {description} 全部通过")
        return True

    except Exception as e:
        print(f"\n[FAIL] {description} 出现错误：{e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("   AI 算法速通——10 个核心算法一键演示")
    print("=" * 60)
    print()
    print("涵盖算法：")
    print("  机器学习: K-Means / KNN / 梯度下降(线性回归) / SMOTE")
    print("  图算法:   PageRank / 二分图判断 / Kruskal最小生成树")
    print("  深度学习: 决策树(信息增益) / 卷积 / Max Pooling")

    results = []

    results.append(run_module(
        "machine_learning",
        "机器学习算法：K-Means / KNN / 梯度下降 / SMOTE"
    ))
    results.append(run_module(
        "graph_algorithms",
        "图算法：PageRank / 二分图判断(BFS染色) / Kruskal"
    ))
    results.append(run_module(
        "dl_basics",
        "深度学习基础：决策树信息增益 / 卷积 / Max Pooling"
    ))

    # 汇总报告
    print(f"\n{'=' * 60}")
    print("  运行结果汇总")
    print(f"{'=' * 60}")
    passed = sum(results)
    total = len(results)
    status = "全部通过" if passed == total else f"{passed}/{total} 通过"
    print(f"  {status}")

    if passed < total:
        print("\n  提示：请检查上方错误信息，确认 numpy 已安装：")
        print("    pip install numpy")
        sys.exit(1)
    else:
        print("\n  所有算法运行正常！可以放心备考了。")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
