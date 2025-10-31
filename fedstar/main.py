import os
import torch
# import torch_geometric.transforms # 不再需要 RandomLinkSplit，因此可以移除
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
import numpy as np  # 新增导入
import random  # 新增导入

from client import Client
from server import Server
from model.graphsage import GraphSAGE
from model.resmlp import ResMLP
from model.Structure_Encoder import DegreeEncoder, RWEncoder


# =============================
# 工具函数
# =============================

# -----------------------------
# 【新增】设置随机种子的逻辑
# -----------------------------
def set_seed(seed):
    """设置所有必要的随机种子以确保实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 针对多GPU
        # 推荐设置，以确保确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_client_data(data: Data, val_ratio=0.2, test_ratio=0.2, device="cuda"):
    """
    对单个图数据划分 train/val/test 节点 Masks。
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
        raise ValueError("训练集大小为零或负数，请检查划分比例。")

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


# -----------------------------
# 结构 encoder 输出维度计算 (保持不变)
# -----------------------------
def get_struct_encoder_out_dim(encoder_cls, params):
    if encoder_cls == DegreeEncoder:
        mode = params.get("mode", "onehot")
        if mode == "onehot":
            return params["max_degree"]
        elif mode == "embed":
            return params["emb_dim"]
        else:  # numeric
            return 1
    elif encoder_cls == RWEncoder:
        return params["num_steps"] + (1 if params.get("add_identity", False) else 0)
    else:
        raise ValueError(f"Unknown encoder class: {encoder_cls}")


# -----------------------------
# 初始化客户端 (保持不变)
# -----------------------------
def load_clients(data_paths, encoder_params, decoder_params, struct_encoder_params, training_params, device,
                 num_classes):
    clients = []

    struct_out_dim_total = 0
    for encoder_cls, params in struct_encoder_params:
        struct_out_dim_total += get_struct_encoder_out_dim(encoder_cls, params)

    decoder_in_dim = encoder_params["output_dim"] + struct_out_dim_total

    for client_id, path in enumerate(data_paths):
        raw_data = torch.load(path)
        data = split_client_data(raw_data, device=device)

        feature_encoder = GraphSAGE(**encoder_params)

        structure_encoders = []
        for encoder_cls, params in struct_encoder_params:
            enc = encoder_cls(**params)
            structure_encoders.append(enc)

        decoder = ResMLP(input_dim=decoder_in_dim, output_dim=num_classes, **decoder_params)

        client = Client(
            client_id=client_id,
            data=data,
            feature_encoder=feature_encoder,
            structure_encoders=structure_encoders,
            decoder=decoder,
            device=device,
            lr=training_params["lr"],
            weight_decay=training_params["weight_decay"],
        )
        clients.append(client)
    return clients


# -----------------------------
# 全局模型评估 (保持不变)
# -----------------------------
def evaluate_global_model(server, clients, use_test=False):
    metrics = []
    for client in clients:
        acc, recall, precision, f1 = client.evaluate(use_test=use_test)
        metrics.append((acc, recall, precision, f1))
        print(f"[Client {client.client_id}] Acc={acc:.4f}, Recall={recall:.4f}, "
              f"Prec={precision:.4f}, F1={f1:.4f}")

    avg = torch.tensor(metrics).mean(dim=0).tolist()
    print(f"\n===> Global Avg: Acc={avg[0]:.4f}, Recall={avg[1]:.4f}, "
          f"Prec={avg[2]:.4f}, F1={avg[3]:.4f}")
    return avg


# =============================
# 主函数
# =============================
def main():
    # 【新增】设置随机种子
    RANDOM_SEED = 42
    set_seed(RANDOM_SEED)

    # -------- Step 1: 参数配置 --------
    data_dir = "../parsed_dataset/cs"
    pyg_data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])

    if not pyg_data_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到任何 PyG 数据文件。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确定输入维度和类别数
    initial_data = torch.load(pyg_data_files[0])
    input_dim = initial_data.x.shape[1]

    num_classes = initial_data.y.max().item() + 1

    print(f"数据输入维度: {input_dim}, 目标类别数: {num_classes}")

    encoder_params = {
        "input_dim": input_dim,
        "hidden_dim": 128,
        "output_dim": 64,
        "num_layers": 3,
        "dropout": 0.4,
    }
    decoder_params = {"hidden_dim": 128, "num_layers": 3, "dropout": 0.3}

    # 每个结构 encoder 独立参数
    struct_encoder_params = [
        (DegreeEncoder, {"max_degree": 50, "mode": "onehot"}),
        (RWEncoder, {"num_steps": 10, "add_identity": False}),
    ]

    training_params = {"lr": 0.001, "weight_decay": 1e-4, "local_epochs": 5}
    num_rounds = 100
    eval_interval = 1

    # -------- Step 2: 初始化客户端 & 全局模型 --------

    struct_out_dim_total = 0
    for encoder_cls, params in struct_encoder_params:
        struct_out_dim_total += get_struct_encoder_out_dim(encoder_cls, params)

    decoder_in_dim = encoder_params["output_dim"] + struct_out_dim_total

    clients = load_clients(
        data_paths=pyg_data_files,
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        struct_encoder_params=struct_encoder_params,
        training_params=training_params,
        device=device,
        num_classes=num_classes
    )

    # 初始化全局模型
    init_params = clients[0].get_parameters()

    structure_encoders_global = []
    for encoder_cls, params in struct_encoder_params:
        enc = encoder_cls(**params)
        structure_encoders_global.append(enc)

    server = Server(
        feature_encoder=GraphSAGE(**encoder_params),
        structure_encoders=structure_encoders_global,
        decoder=ResMLP(input_dim=decoder_in_dim,
                       output_dim=num_classes,
                       **decoder_params),
        device=device,
    )
    server.set_global_parameters(init_params)

    # -------- Step 3: 联邦训练循环 --------
    print("\n================ Federated Training Start ================")
    best_f1 = -1
    best_state = None

    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")

        for client in clients:
            client.local_train(training_params["local_epochs"])
            print(f"Client {client.client_id} finished local training.")

        server.aggregate_all_weights(clients)
        server.distribute_parameters(clients)

        if rnd % eval_interval == 0:
            avg_acc, avg_recall, avg_prec, avg_f1 = evaluate_global_model(server, clients, use_test=False)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_state = server.get_global_parameters()
                print("===> New best model saved")

    # -------- Step 4: 最终评估 --------
    print("\n================ Federated Training Finished ================")
    if best_state is not None:
        server.set_global_parameters(best_state)
        server.distribute_parameters(clients)

    print("\n================ Final Evaluation ================")
    evaluate_global_model(server, clients, use_test=True)


if __name__ == "__main__":
    main()
