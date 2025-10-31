import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
from client import Client
# 导入您新的模型文件
from model.gcn import GCN
from model.graphsage import GraphSAGE
from model.resmlp import ResMLP
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


# --- 1. 数据划分逻辑（保持不变） ---
def split_client_data_for_node_classification(data: Data, val_ratio=0.2, test_ratio=0.2):
    """
    根据给定的比例，为客户端数据生成 train/val/test 节点 Masks。
    """
    num_nodes = data.num_nodes
    num_labeled_nodes = num_nodes

    indices = torch.randperm(num_labeled_nodes)

    num_test = int(test_ratio * num_labeled_nodes)
    num_val = int(val_ratio * num_labeled_nodes)
    num_train = num_labeled_nodes - num_test - num_val

    if num_train <= 0:
        raise ValueError("训练集大小为零或负数，请检查划分比例。")

    test_indices = indices[:num_test]
    val_indices = indices[num_test:num_test + num_val]
    train_indices = indices[num_test + num_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    data.edge_index = to_undirected(data.edge_index, num_nodes=num_nodes)

    return data


# --- 2. 客户端加载逻辑（适配 FedProto） ---
def load_all_clients(pyg_data_paths, encoder_params, classifier_params, training_params, device, lambda_proto,
                     num_classes, prototype_dim):  # <--- 【修改点】新增 prototype_dim
    clients = []

    print(f"检测到类别总数 (Output Dim): {num_classes}")
    print(f"原型维度 (Prototype Dim): {prototype_dim}")

    for client_id, path in enumerate(pyg_data_paths):
        raw_data = torch.load(path)

        data = split_client_data_for_node_classification(
            raw_data,
            val_ratio=0.2,
            test_ratio=0.2
        )
        data = data.to(device)

        # 编码器实例化
        encoder = GraphSAGE(
            input_dim=encoder_params['input_dim'],
            hidden_dim=encoder_params['hidden_dim'],
            output_dim=prototype_dim,
            num_layers=encoder_params['num_layers'],
            dropout=encoder_params['dropout']
        )

        # 分类器实例化
        classifier = ResMLP(
            input_dim=prototype_dim,
            hidden_dim=classifier_params['hidden_dim'],
            output_dim=num_classes,
            num_layers=classifier_params['num_layers'],
            dropout=classifier_params['dropout']
        )

        # 【修改点】实例化 Client 时传入 prototype_dim
        client = Client(
            client_id=client_id,
            data=data,
            encoder=encoder,
            classifier=classifier,
            device=device,
            lr=training_params['lr'],
            weight_decay=training_params['weight_decay'],
            lambda_proto=lambda_proto,
            num_classes=num_classes,
            prototype_dim=prototype_dim  # <--- 传入维度
        )
        clients.append(client)

    return clients, num_classes


# --- 3. FedAvg 聚合逻辑（用于模型权重） ---
def average_state_dicts(state_dicts):
    """标准 FedAvg 权重平均"""
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state


# --- 4. FedProto 核心：原型聚合逻辑 ---
def aggregate_prototypes(clients, num_classes, prototype_dim, device):
    """
    聚合所有客户端上传的本地原型，计算加权平均的全局原型。

    返回: global_prototypes (torch.Tensor of shape [num_classes, prototype_dim])
    """
    # 初始化全局原型和计数器
    global_prototypes = torch.zeros(num_classes, prototype_dim).to(device)
    global_counts = torch.zeros(num_classes).to(device)

    # 1. 收集所有客户端的本地原型和计数
    # 注意：get_local_prototypes 在 client.py 中已实现，是在评估模式下进行的无梯度计算
    all_local_protos_and_counts = []
    for client in clients:
        # get_local_prototypes 返回 (dict: {class_id: proto}, list: [(class_id, count)])
        local_protos, proto_counts = client.get_local_prototypes()
        all_local_protos_and_counts.append((local_protos, proto_counts))

    # 2. 遍历所有客户端，执行加权聚合
    for local_protos, proto_counts in all_local_protos_and_counts:
        for class_id, count in proto_counts:
            prototype = local_protos[class_id].to(device)

            # 累加：原型 * 样本数
            global_prototypes[class_id] += prototype * count
            # 累加：样本数
            global_counts[class_id] += count

    # 3. 计算加权平均
    for class_id in range(num_classes):
        count = global_counts[class_id]
        if count > 0:
            global_prototypes[class_id] /= count
        # 否则，该类别没有样本，原型保持零向量

    return global_prototypes


# --- 5. 评估逻辑（保持不变） ---
def evaluate_all_clients(clients, use_test=False):
    """评估所有客户端模型并计算平均指标"""
    metrics = []
    for client in clients:
        acc, recall, precision, f1 = client.evaluate(use_test=use_test)
        metrics.append((acc, recall, precision, f1))
        print(f"Client {client.client_id}: Acc={acc:.4f}, Recall={recall:.4f}, "
              f"Prec={precision:.4f}, F1={f1:.4f}")

    if metrics:
        avg_metrics = torch.tensor(metrics).mean(dim=0).tolist()
        print(f"\n===> Average Metrics: Acc={avg_metrics[0]:.4f}, Recall={avg_metrics[1]:.4f}, "
              f"Prec={avg_metrics[2]:.4f}, F1={avg_metrics[3]:.4f}")
        return avg_metrics
    return [0.0, 0.0, 0.0, 0.0]


# --- 6. 主训练流程 ---
if __name__ == "__main__":

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    RANDOM_SEED = 42
    set_seed(RANDOM_SEED)

    # 1. 配置路径与参数
    data_dir = "../parsed_dataset/cs"
    pyg_data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])

    if not pyg_data_files:
        print(f"❌ 错误: 未找到任何 PyG 子图文件在 {data_dir}。请检查路径和文件后缀。")
        exit()

    # 读取第一个文件来确定输入维度和类别数
    initial_data = torch.load(pyg_data_files[0])
    initial_x = initial_data.x
    initial_y = initial_data.y

    input_dim = initial_x.shape[1] if initial_x is not None and initial_x.dim() > 0 else 1
    # 确保类别数计算基于有效的标签
    num_classes_calc = initial_y.max().item() + 1 if initial_y is not None and initial_y.numel() > 0 and initial_y.max().item() >= 0 else 7

    # 关键修改 A: FedProto 参数
    LAMBDA_PROTO = 1.0  # <--- 设置原型损失正则化权重。

    encoder_params = {
        'input_dim': input_dim,
        'hidden_dim': 128,
        'output_dim': 64,  # 原型维度 (D_emb)
        'num_layers': 3,
        'dropout': 0.5
    }

    classifier_params = {
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.3
    }

    training_params = {
        'lr': 0.001,
        'weight_decay': 1e-4,
        'local_epochs': 5
    }

    num_rounds = 600
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取原型维度 (所有客户端共享)
    prototype_dim = encoder_params['output_dim']

    # 2. 初始化客户端
    # 【修改点】调用 load_all_clients 时传入 prototype_dim
    clients, num_classes = load_all_clients(
        pyg_data_files,
        encoder_params,
        classifier_params,
        training_params,
        device,
        LAMBDA_PROTO,
        num_classes_calc,
        prototype_dim
    )

    # 初始全局原型：全零向量
    global_prototypes = torch.zeros(num_classes, prototype_dim).to(device)

    # 首次同步全局原型（全零）
    for client in clients:
        client.set_global_prototypes(global_prototypes)

    best_f1 = -1
    best_encoder_state = None
    best_classifier_state = None
    best_prototypes = None  # 保存最佳原型状态

    print(f"\n🚀 Federated Algorithm: FedProto (Lambda={LAMBDA_PROTO})")
    print("================ Federated Training Start ================\n")

    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")

        # 3. 每个客户端本地训练
        for client in clients:
            for _ in range(training_params['local_epochs']):
                loss = client.train()

        # 4. 聚合：模型权重 (FedAvg) + 原型 (FedProto)

        # 4a. 模型权重聚合 (FedAvg 方式)
        encoder_states = [copy.deepcopy(client.get_encoder_state()) for client in clients]
        classifier_states = [copy.deepcopy(client.get_classifier_state()) for client in clients]
        global_encoder_state = average_state_dicts(encoder_states)
        global_classifier_state = average_state_dicts(classifier_states)

        # 4b. 原型聚合 (FedProto 方式)
        global_prototypes = aggregate_prototypes(clients, num_classes, prototype_dim, device)

        # 5. 同步参数
        for client in clients:
            # 同步模型权重
            client.set_encoder_state(global_encoder_state)
            client.set_classifier_state(global_classifier_state)
            # 同步全局原型
            client.set_global_prototypes(global_prototypes)

        # 6. 联邦评估
        avg_acc, avg_recall, avg_prec, avg_f1 = evaluate_all_clients(clients, use_test=False)

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_encoder_state = global_encoder_state
            best_classifier_state = global_classifier_state
            best_prototypes = global_prototypes
            print("===> New best global model and prototypes saved.")

    print("\n================ Federated Training Finished ================\n")

    # 7. 最终模型评估
    for client in clients:
        client.set_encoder_state(best_encoder_state)
        client.set_classifier_state(best_classifier_state)
        client.set_global_prototypes(best_prototypes)

    print("\n================ Final Evaluation on Test Set ================")
    evaluate_all_clients(clients, use_test=True)

    print("---------------------------------------------------------------")
    print(f"最终最佳 F1 Score (Validation): {best_f1:.4f}")
    print("---------------------------------------------------------------")
