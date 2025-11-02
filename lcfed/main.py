import os
import torch
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
import numpy as np
import random
import copy
import torch.nn as nn

# 导入 LCFed 模块
from client import LcClient
from server import LcServer
# 导入模型 (我们将 GNN 和 ResMLP 封装在一个容器模型中，作为完整的 theta)
from model.graphsage import GraphSAGE
from model.resmlp import ResMLP
# from model.gnn_classifier import GNNClassifier  # 假设我们有一个组合模型


# =============================
# 工具函数
# =============================
def set_seed(seed):
    """设置所有必要的随机种子以确保实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_client_data(data: Data, val_ratio=0.2, test_ratio=0.2, device="cuda"):
    """
    对单个图数据划分 train/val/test 节点 Masks。（节点分类版本）
    """
    data = data.to(device)
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    num_nodes = data.num_nodes
    indices = torch.nonzero(data.y >= 0, as_tuple=False).flatten()
    num_labeled_nodes = indices.size(0)

    perm = torch.randperm(num_labeled_nodes)
    indices = indices[perm]

    num_test = int(test_ratio * num_labeled_nodes)
    num_val = int(val_ratio * num_labeled_nodes)
    num_train = num_labeled_nodes - num_test - num_val

    if num_train <= 0:
        print(f"警告：客户端训练集大小为零或负数（{num_train}）。")
        num_val = int(num_labeled_nodes * 0.1)
        num_train = num_labeled_nodes - num_test - num_val

    test_indices = indices[:num_test]
    val_indices = indices[num_test:num_test + num_val]
    train_indices = indices[num_test + num_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def initialize_lcfed_model(input_dim, num_classes, encoder_params, decoder_params, device):
    """
    初始化一个完整的全局模型 theta_Global (GNN + Classifier)。
    """
    # 假设 GNNClassifier 是一个包含 GNN 和 ResMLP 的 nn.Module
    # GNNClassifier 需要在 Model/ 文件夹中定义 (如果还未定义，请使用 GNN 和 ResMLP 的串联)

    # 1. 创建 GNN 作为特征提取器
    feature_extractor = GraphSAGE(**encoder_params)
    feature_dim = encoder_params["output_dim"]

    # 2. 创建 ResMLP 作为分类器
    classifier = ResMLP(
        input_dim=feature_dim,
        output_dim=num_classes,
        **decoder_params
    )

    # 3. 将两者组合成一个 nn.Sequential 或自定义 nn.Module
    # 为了简化，这里返回一个包含两者逻辑的 nn.Module（例如 GNNClassifier）

    # 由于我们没有 GNNClassifier.py 文件，这里用一个 lambda 函数模拟其前向传播
    # 注意：在实际 PyTorch 代码中，这应该是一个继承 nn.Module 的类
    class CombinedModel(nn.Module):
        def __init__(self, fe, cl):
            super().__init__()
            self.fe = fe
            self.cl = cl

        def forward(self, x, edge_index):
            z = self.fe(x, edge_index)
            return self.cl(z)

    global_model = CombinedModel(feature_extractor, classifier).to(device)

    return global_model


def load_clients(data_paths, base_global_model, training_params, device):
    """
    初始化客户端，并将全局模型下发给每个客户端作为初始模型。
    """
    clients = []

    for client_id, path in enumerate(data_paths):
        raw_data = torch.load(path)
        data = split_client_data(raw_data, device=device)

        # 客户端模型 theta_i 是全局模型的一个深拷贝
        client_model = copy.deepcopy(base_global_model).to(device)

        client = LcClient(
            client_id=client_id,
            data=data,
            base_model=client_model,
            device=device,
            lr=training_params["lr"],
            weight_decay=training_params["weight_decay"],
        )
        clients.append(client)
    return clients


def evaluate_global_model(clients, use_test=False):
    """
    全局模型评估：评估每个客户端的个性化模型 (theta_i)，并计算平均性能。
    """
    metrics = []
    # 记录客户端及其使用的集群 ID
    client_infos = []

    for client in clients:
        acc, recall, precision, f1 = client.evaluate(use_test=use_test)
        metrics.append((acc, recall, precision, f1))
        client_infos.append(f"Client {client.client_id} (Cluster {client.current_cluster_id})")

    # 打印客户端信息
    print("Clients evaluated:", ", ".join(client_infos))

    avg = torch.tensor(metrics).mean(dim=0).tolist()
    print(f"===> Global Avg: Acc={avg[0]:.4f}, Recall={avg[1]:.4f}, "
          f"Prec={avg[2]:.4f}, F1={avg[3]:.4f}")
    return avg


# =============================
# 主函数
# =============================
def main():
    # --- 配置 ---
    RANDOM_SEED = 42
    set_seed(RANDOM_SEED)

    NUM_CLUSTERS = 3  # LCFed 的集群数量 K

    data_dir = "../parsed_dataset/cs"
    pyg_data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])

    if not pyg_data_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到任何 PyG 数据文件。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确定输入维度和类别数
    initial_data = torch.load(pyg_data_files[0])
    input_dim = initial_data.x.shape[1]
    num_classes = initial_data.y.max().item() + 1

    print(f"数据输入维度: {input_dim}, 目标类别数: {num_classes}, 集群数 K: {NUM_CLUSTERS}")

    # 模型参数
    encoder_params = {
        "input_dim": input_dim,
        "hidden_dim": 128,
        "output_dim": 64,
        "num_layers": 3,
        "dropout": 0.4,
    }
    decoder_params = {"hidden_dim": 128, "num_layers": 3, "dropout": 0.3}

    training_params = {"lr": 0.001, "weight_decay": 1e-4, "local_epochs": 5}

    num_rounds = 600
    eval_interval = 1

    # -------- Step 1: 初始化模型 & 服务器 --------
    global_model = initialize_lcfed_model(input_dim, num_classes, encoder_params, decoder_params, device)

    # 初始化服务器
    server = LcServer(
        global_model=global_model,
        device=device,
        num_clusters=NUM_CLUSTERS
    )

    # 初始化客户端
    clients = load_clients(
        data_paths=pyg_data_files,
        base_global_model=global_model,
        training_params=training_params,
        device=device,
    )

    # -------- Step 2: LCFed 联邦训练循环 --------
    print("\n================ LCFed Federated Training Start ================")
    best_f1 = -1
    best_state = None

    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")

        # 1. Server 分发全局模型 theta_Global 和 集群偏差 Delta_theta_Cluster, k
        server.distribute_parameters(clients)

        # 2. Client 本地训练 (theta_i) 并上传更新 (Delta_theta_i 和 theta_i)
        client_updates = []
        client_ids = []
        print("Client Step: Local training theta_i = theta_Global + Delta_theta_Cluster, k...")
        for client in clients:
            client.local_train(training_params["local_epochs"])
            update = client.get_trained_update()
            client_updates.append(update)
            client_ids.append(client.client_id)
            print(f"Client {client.client_id} finished local training.")

        # 3. Server 聚合与聚类
        print("Server Step: Aggregating theta_Global, clustering Delta_theta_i, and aggregating cluster biases...")

        # a. 聚合全局模型 theta_Global (FedAvg of theta_i)
        server.aggregate_global_model(client_updates)

        # b. 对个性化偏差 Delta_theta_i 进行 K-means 聚类
        server.cluster_personalized_bias(client_updates, client_ids)

        # c. 聚合集群偏差 Delta_theta_Cluster, k
        server.aggregate_cluster_bias(client_updates, client_ids)

        # 4. 评估 (客户端使用本地个性化模型 theta_i 评估)
        if rnd % eval_interval == 0:
            print("\nEvaluation (Validation Set):")
            avg_acc, avg_recall, avg_prec, avg_f1 = evaluate_global_model(clients, use_test=False)

            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_state = server.get_global_parameters()
                print("===> New best LCFed state saved")

    # -------- Step 3: 最终评估 --------
    print("\n================ LCFed Federated Training Finished ================")
    if best_state is not None:
        # 将 Server 恢复到最佳状态，并分发给 Client 以进行最终评估
        server.set_global_parameters(best_state)
        server.distribute_parameters(clients)

    print("\n================ Final Evaluation (Test Set) ================")
    evaluate_global_model(clients, use_test=True)


if __name__ == "__main__":
    main()