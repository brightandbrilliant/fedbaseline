import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
from client import Client  # 假设这是上一步我们创建的Client.py
# 导入您新的模型文件
from model.gcn import GCN
from model.graphsage import GraphSAGE
from model.resmlp import ResMLP  # 导入新的分类器
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


# --- 1. 数据划分逻辑（适配节点分类） ---
def split_client_data_for_node_classification(data: Data, val_ratio=0.2, test_ratio=0.2):
    """
    根据给定的比例，为客户端数据生成 train/val/test 节点 Masks。

    注意：此函数假设所有节点的标签 (data.y) 都是有效的（即不是-1）。
    如果 Citeseer 等数据集的原始标签存在 -1，需要提前过滤或处理。
    """
    num_nodes = data.num_nodes
    num_labeled_nodes = num_nodes  # 假设所有节点都参与划分

    # 随机化节点索引
    indices = torch.randperm(num_labeled_nodes)

    # 计算划分大小
    num_test = int(test_ratio * num_labeled_nodes)
    num_val = int(val_ratio * num_labeled_nodes)
    num_train = num_labeled_nodes - num_test - num_val

    # 确保划分合理
    if num_train <= 0:
        raise ValueError("训练集大小为零或负数，请检查划分比例。")

    # 分割索引
    test_indices = indices[:num_test]
    val_indices = indices[num_test:num_test + num_val]
    train_indices = indices[num_test + num_val:]

    # 创建 Masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # 附加到数据对象上 (这是关键步骤)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # 确保边是无向的，用于GNN
    data.edge_index = to_undirected(data.edge_index, num_nodes=num_nodes)

    return data


# --- 2. 客户端加载逻辑（适配节点分类模型） ---
def load_all_clients(pyg_data_paths, encoder_params, classifier_params, training_params, device):
    clients = []

    # 获取类别数 (Node Classification Output Dim)
    # 假设第一个文件的标签反映了总类别数
    initial_data = torch.load(pyg_data_paths[0])
    num_classes = initial_data.y.max().item() + 1

    print(f"检测到类别总数 (Output Dim): {num_classes}")

    for client_id, path in enumerate(pyg_data_paths):
        raw_data = torch.load(path)

        # 关键修改 1: 调用节点分类的划分函数
        data = split_client_data_for_node_classification(
            raw_data,
            val_ratio=0.2,
            test_ratio=0.2
        )
        data = data.to(device)

        # 编码器实例化（使用GraphSAGE作为例子）
        encoder = GraphSAGE(
            input_dim=encoder_params['input_dim'],
            hidden_dim=encoder_params['hidden_dim'],
            output_dim=encoder_params['output_dim'],  # 编码器输出嵌入维度
            num_layers=encoder_params['num_layers'],
            dropout=encoder_params['dropout']
        )

        # 关键修改 2: 实例化 ResMLPClassifier，输出维度为类别数
        classifier = ResMLP(
            input_dim=encoder_params['output_dim'],  # 解码器输入 = 编码器输出
            hidden_dim=classifier_params['hidden_dim'],
            output_dim=num_classes,  # 解码器输出 = 类别数
            num_layers=classifier_params['num_layers'],
            dropout=classifier_params['dropout']
        )

        # 关键修改 3: 实例化 Client 时传入 'classifier'
        client = Client(
            client_id=client_id,
            data=data,
            encoder=encoder,
            classifier=classifier,  # 传入 classifier
            device=device,
            lr=training_params['lr'],
            weight_decay=training_params['weight_decay']
        )
        clients.append(client)

    return clients, num_classes


# --- 3. FedAvg 聚合逻辑（完全复用） ---
def average_state_dicts(state_dicts):
    """标准 FedAvg 权重平均"""
    avg_state = {}
    for key in state_dicts[0].keys():
        # 将所有客户端的同一层权重堆叠起来，然后求平均
        avg_state[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state


# --- 4. 评估逻辑（完全复用） ---
def evaluate_all_clients(clients, use_test=False):
    """评估所有客户端模型并计算平均指标"""
    metrics = []
    for client in clients:
        # client.evaluate() 现在返回节点分类指标
        acc, recall, precision, f1 = client.evaluate(use_test=use_test)
        metrics.append((acc, recall, precision, f1))
        print(f"Client {client.client_id}: Acc={acc:.4f}, Recall={recall:.4f}, "
              f"Prec={precision:.4f}, F1={f1:.4f}")

    # 计算平均值
    if metrics:
        avg_metrics = torch.tensor(metrics).mean(dim=0).tolist()
        print(f"\n===> Average Metrics: Acc={avg_metrics[0]:.4f}, Recall={avg_metrics[1]:.4f}, "
              f"Prec={avg_metrics[2]:.4f}, F1={avg_metrics[3]:.4f}")
        return avg_metrics
    return [0.0, 0.0, 0.0, 0.0]  # 避免空列表错误


# --- 5. 主训练流程 ---
if __name__ == "__main__":

    # --- 全局随机种子固定 (可选，但推荐) ---
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
    # 假设您的子图文件位于这个路径下
    data_dir = "../parsed_dataset/cs"  # 替换为您的子图保存目录
    pyg_data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])

    if not pyg_data_files:
        print(f"❌ 错误: 未找到任何 PyG 子图文件在 {data_dir}。请检查路径和文件后缀。")
        exit()

    initial_x = torch.load(pyg_data_files[0]).x
    if initial_x is None or initial_x.dim() == 0:
        # 容错处理：如果特征是None或空tensor，给一个默认值
        input_dim = 1
        print("⚠️ 警告: 无法读取第一个文件的特征维度，设置为默认值 1。")
    else:
        input_dim = initial_x.shape[1]

    encoder_params = {
        'input_dim': input_dim,
        'hidden_dim': 128,
        'output_dim': 64,  # GNN编码器的输出嵌入维度 D_emb
        'num_layers': 3,
        'dropout': 0.5
    }

    # 关键修改: 将 decoder_params 命名为 classifier_params
    classifier_params = {
        'hidden_dim': 128,
        'num_layers': 3,  # 分类头层数不宜太深
        'dropout': 0.3
    }

    training_params = {
        'lr': 0.001,
        'weight_decay': 1e-4,
        'local_epochs': 5
    }

    num_rounds = 600
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 初始化客户端
    # 关键修改：load_all_clients 现在返回 clients 和 num_classes
    clients, num_classes = load_all_clients(pyg_data_files, encoder_params, classifier_params, training_params, device)

    best_f1 = -1  # 继续追踪F1 Score
    best_encoder_state = None
    best_classifier_states = []  # 关键修改：重命名最佳状态变量

    print("\n================ Federated Training Start ================\n")
    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")

        # 3. 每个客户端本地训练
        for client in clients:
            for _ in range(training_params['local_epochs']):
                loss = client.train()

        # 4. FedAvg 聚合参数
        encoder_states = [copy.deepcopy(client.get_encoder_state()) for client in clients]
        classifier_states = [copy.deepcopy(client.get_classifier_state()) for client in clients]

        global_encoder_state = average_state_dicts(encoder_states)

        # 5. 同步参数
        for client in clients:
            client.set_encoder_state(global_encoder_state)

        # 6. 联邦评估
        avg_acc, avg_recall, avg_prec, avg_f1 = evaluate_all_clients(clients, use_test=False)

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_encoder_state = global_encoder_state
            best_classifier_states = classifier_states
            print("===> New best global model saved.")

    print("\n================ Federated Training Finished ================\n")

    cnt = 0
    # 7. 最终模型评估
    for client in clients:
        client.set_encoder_state(best_encoder_state)
        client.set_classifier_state(best_classifier_states[cnt])
        cnt += 1

    print("\n================ Final Evaluation on Test Set ================")
    evaluate_all_clients(clients, use_test=True)

    print("---------------------------------------------------------------")
    print(f"最终最佳 F1 Score (Validation): {best_f1:.4f}")
    print("---------------------------------------------------------------")
