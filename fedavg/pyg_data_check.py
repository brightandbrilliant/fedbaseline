import os
import torch
from torch_geometric.data import Data
from collections import Counter
import numpy as np

# --- 配置 ---
# 假设您的客户端数据子图文件位于这个路径下
DATA_DIR = "../parsed_dataset/cs"
PYG_DATA_FILES = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pt")])


def get_label_distribution(y: torch.Tensor, mask: torch.Tensor = None) -> Counter:
    """获取指定 Mask 下的标签分布。"""
    if mask is not None:
        y_masked = y[mask]
    else:
        y_masked = y

    # 过滤掉无效标签（如 -1）
    valid_labels = y_masked[y_masked >= 0].cpu().tolist()

    return Counter(valid_labels)


def analyze_client_data(file_path: str, client_id: int):
    """加载单个客户端数据并打印详细统计信息。"""
    print(f"\n--- 📊 客户端 {client_id} 统计信息 ({os.path.basename(file_path)}) ---")

    try:
        data: Data = torch.load(file_path)
    except Exception as e:
        print(f"❌ 错误: 无法加载文件 {file_path}. 错误: {e}")
        return

    # 1. 基础信息
    print(f"  - 节点总数 (N): {data.num_nodes}")
    print(f"  - 边总数 (E): {data.num_edges}")

    # 2. 特征和标签信息
    if data.x is not None:
        print(f"  - 节点特征维度 (D_x): {data.x.shape[1]}")
    if data.y is not None:
        # 计算所有节点的有效标签数 (>= 0)
        num_valid_labels = (data.y >= 0).sum().item()
        print(f"  - 节点标签总数: {data.y.numel()}")
        print(f"  - 有效标签节点数: {num_valid_labels}")

        # 获取所有有效标签的分布
        full_distribution = get_label_distribution(data.y)
        total_valid_nodes = sum(full_distribution.values())
        print(f"  - 标签分布 (所有有效节点): {total_valid_nodes} 个节点")
        for label, count in sorted(full_distribution.items()):
            print(f"    - 类别 {label}: {count} ({count / total_valid_nodes:.2%})")

    # 3. 训练/验证/测试划分信息
    has_masks = hasattr(data, 'train_mask') and data.train_mask is not None

    if has_masks:
        train_size = data.train_mask.sum().item()
        val_size = data.val_mask.sum().item()
        test_size = data.test_mask.sum().item()

        total_masked = train_size + val_size + test_size

        print(f"\n  --- 划分统计 (基于 Mask) ---")
        print(f"  - 训练集节点数: {train_size}")
        print(f"  - 验证集节点数: {val_size}")
        print(f"  - 测试集节点数: {test_size}")
        print(f"  - 划分总节点数: {total_masked}")

        # 4. 训练集标签分布 (Non-IID 分析核心)
        if data.y is not None and train_size > 0:
            train_distribution = get_label_distribution(data.y, data.train_mask)
            total_train_valid = sum(train_distribution.values())

            print(f"\n  --- 训练集标签分布 (N_{{train}} = {total_train_valid}) ---")
            for label, count in sorted(train_distribution.items()):
                print(f"    - 类别 {label}: {count} ({count / total_train_valid:.2%})")

    else:
        print("\n  --- 划分统计 ---")
        print("  ⚠️ 警告: 客户端数据中未找到 train/val/test mask。")


def main():
    """主函数，遍历所有客户端文件并进行分析。"""
    print(f"✨ 正在分析客户端数据集目录: {DATA_DIR}")

    if not os.path.isdir(DATA_DIR):
        print(f"❌ 错误: 目录 {DATA_DIR} 不存在。请检查配置。")
        return

    if not PYG_DATA_FILES:
        print(f"❌ 错误: 目录中未找到任何 .pt 文件。")
        return

    # 全局统计 (可选，但很有用)
    all_client_nodes = []

    for client_id, file_path in enumerate(PYG_DATA_FILES):
        analyze_client_data(file_path, client_id)

        try:
            data: Data = torch.load(file_path)
            all_client_nodes.append(data.num_nodes)
        except:
            pass

    # 打印全局汇总信息
    if all_client_nodes:
        total_nodes = sum(all_client_nodes)
        min_nodes = min(all_client_nodes)
        max_nodes = max(all_client_nodes)
        avg_nodes = np.mean(all_client_nodes)

        print("\n==================== 📊 全局汇总 ====================")
        print(f"  - 客户端总数: {len(PYG_DATA_FILES)}")
        print(f"  - 所有客户端节点总数: {total_nodes}")
        print(f"  - 客户端节点数范围: {min_nodes} (Min) ~ {max_nodes} (Max)")
        print(f"  - 客户端平均节点数: {avg_nodes:.2f}")
        print("======================================================")


if __name__ == "__main__":
    main()