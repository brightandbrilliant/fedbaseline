import os
import torch
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
import numpy as np
import random
import copy

from client import Client
from server import Server
from model.graphsage import GraphSAGE
from model.resmlp import ResMLP



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
        # 简单修正，确保训练集不为空
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


def initialize_cluster_models(encoder_params, decoder_params, num_clusters, num_classes, device):
    """
    初始化 K 组集群模型。
    """
    # 【修改】输入维度：现在 decoder 的输入维度就是 GNN 的输出维度
    decoder_in_dim = encoder_params["output_dim"]

    feature_encoders_k = []
    decoders_k = []

    # 初始化 K 组模型，并使用第一组的模型作为所有 K 组模型的初始值
    base_feature_encoder = GraphSAGE(**encoder_params).to(device)
    base_decoder = ResMLP(input_dim=decoder_in_dim, output_dim=num_classes, **decoder_params).to(device)

    for k in range(num_clusters):
        # 使用深拷贝确保 K 个模型是独立的实例，且拥有相同的初始权重
        feature_encoders_k.append(copy.deepcopy(base_feature_encoder))
        decoders_k.append(copy.deepcopy(base_decoder))

    return feature_encoders_k, decoders_k


def load_clients(data_paths, feature_encoders_k_base, decoders_k_base, training_params, device, num_clusters):
    """
    初始化客户端，并将 K 组模型下发给每个客户端。
    """
    clients = []

    for client_id, path in enumerate(data_paths):
        raw_data = torch.load(path)
        data = split_client_data(raw_data, device=device)

        # 客户端接收 K 组模型（深拷贝保证独立性）
        feature_encoders_k = [copy.deepcopy(enc).to(device) for enc in feature_encoders_k_base]
        decoders_k = [copy.deepcopy(dec).to(device) for dec in decoders_k_base]

        # 【修改】不再传入 structure_encoders

        client = Client(
            client_id=client_id,
            data=data,
            feature_encoders=feature_encoders_k,
            decoders=decoders_k,
            device=device,
            lr=training_params["lr"],
            weight_decay=training_params["weight_decay"],
            num_clusters=num_clusters
        )
        clients.append(client)
    return clients


def evaluate_global_model(server, clients, use_test=False):
    """
    全局模型评估：每个客户端选择 K 组模型中表现最好的进行评估。
    """
    metrics = []
    for client in clients:
        # client.evaluate 内部会调用 select_best_cluster
        acc, recall, precision, f1 = client.evaluate(use_test=use_test)
        metrics.append((acc, recall, precision, f1))

    avg = torch.tensor(metrics).mean(dim=0).tolist()
    print(f"===> Global Avg: Acc={avg[0]:.4f}, Recall={avg[1]:.4f}, "
          f"Prec={avg[2]:.4f}, F1={avg[3]:.4f}")
    return avg

def main():
    # --- 配置 ---
    RANDOM_SEED = 42
    NUM_CLUSTERS = 2
    set_seed(RANDOM_SEED)

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

    encoder_params = {
        "input_dim": input_dim,
        "hidden_dim": 128,
        "output_dim": 64,  # GNN 的输出维度
        "num_layers": 3,
        "dropout": 0.4,
    }
    # ResMLP (分类头) 现在直接接受 GNN 的输出
    decoder_params = {"hidden_dim": 128, "num_layers": 3, "dropout": 0.3}

    # 【移除】结构编码器参数定义
    # struct_encoder_params = [...]

    training_params = {"lr": 0.001, "weight_decay": 1e-4, "local_epochs": 5}
    num_rounds = 50
    eval_interval = 5

    # -------- Step 1: 初始化 K 组模型 & 服务器 --------
    # 【修改】不再传入 struct_encoder_params
    feature_encoders_k_base, decoders_k_base = initialize_cluster_models(
        encoder_params, decoder_params, NUM_CLUSTERS, num_classes, device
    )

    # 初始化服务器
    server = Server(
        feature_encoders=feature_encoders_k_base,
        decoders=decoders_k_base,
        device=device,
        num_clusters=NUM_CLUSTERS
    )

    # 初始化客户端
    # 【修改】不再传入 structure_encoders_global
    clients = load_clients(
        data_paths=pyg_data_files,
        feature_encoders_k_base=feature_encoders_k_base,
        decoders_k_base=decoders_k_base,
        training_params=training_params,
        device=device,
        num_clusters=NUM_CLUSTERS
    )

    # -------- Step 2: IFCA 联邦训练循环 --------
    print("\n================ IFCA Federated Training Start ================")
    best_f1 = -1
    best_state = None

    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")

        client_updates = []

        # 1. 聚类分配 (Assignment Step)
        print("Assignment Step: Clients select best cluster...")
        for client in clients:
            client.select_best_cluster()

        # 2. 本地训练和上传 (Optimization Step)
        print("Optimization Step: Clients train locally and upload updates...")
        for client in clients:
            client.local_train(training_params["local_epochs"])
            update = client.get_trained_update()
            client_updates.append(update)
            print(f"Client {client.client_id} finished local training for Cluster {update['cluster_id']}.")

        # 3. 服务器聚合
        server.aggregate_all_weights(client_updates)
        server.distribute_parameters(clients)

        # 4. 评估
        if rnd % eval_interval == 0:
            print("\nEvaluation:")
            avg_acc, avg_recall, avg_prec, avg_f1 = evaluate_global_model(server, clients, use_test=False)

            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_state = server.get_global_parameters()
                print("===> New best K-model state saved")

    # -------- Step 3: 最终评估 --------
    print("\n================ IFCA Federated Training Finished ================")
    if best_state is not None:
        server.set_global_parameters(best_state)
        server.distribute_parameters(clients)

    print("\n================ Final Evaluation ================")
    evaluate_global_model(server, clients, use_test=True)


if __name__ == "__main__":
    main()
