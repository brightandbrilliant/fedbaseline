import os
import torch
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
import numpy as np
import random
import copy

# 导入所有文件
from client import PfedfdaClient
from server import PfedfdaServer
# 导入模型 (假设它们在 Model/ 文件夹或已在路径中)
from model.graphsage import GraphSAGE
from model.resmlp import ResMLP


# from feature_distribution import FeatureDistribution # Client/Server 内部已导入

# =============================
# 工具函数 (与 IFCA/FedSTAR 保持一致)
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


def initialize_pFedFDA_models(encoder_params, decoder_params, num_classes, device):
    """
    初始化共享特征提取器 f (GNN) 和全局分类器 g_G (ResMLP)。
    """
    feature_dim = encoder_params["output_dim"]

    # f: 共享特征提取器 (GNN)
    base_feature_extractor = GraphSAGE(**encoder_params).to(device)

    # g_G: 全局生成式分类器 (ResMLP)，输入维度为特征维度
    base_classifier_global = ResMLP(
        input_dim=feature_dim,
        output_dim=num_classes,
        **decoder_params
    ).to(device)

    return base_feature_extractor, base_classifier_global, feature_dim


def load_clients(data_paths, base_feature_extractor, base_classifier_global, training_params, device, feature_dim,
                 gmm_params):
    """
    初始化客户端，并将 f 和 g_G 下发给每个客户端作为初始模型。
    """
    clients = []

    # 客户端 f_i 和 g_i 必须是 Server 模型 f 和 g_G 的深拷贝副本
    base_f_copy = copy.deepcopy(base_feature_extractor)
    base_g_copy = copy.deepcopy(base_classifier_global)

    for client_id, path in enumerate(data_paths):
        raw_data = torch.load(path)
        data = split_client_data(raw_data, device=device)

        # 客户端模型：f_i 和 g_i
        f_i = copy.deepcopy(base_f_copy).to(device)
        g_i = copy.deepcopy(base_g_copy).to(device)

        client = PfedfdaClient(
            client_id=client_id,
            data=data,
            feature_extractor=f_i,
            classifier=g_i,
            device=device,
            lr=training_params["lr"],
            weight_decay=training_params["weight_decay"],
            feature_dim=feature_dim,
            gmm_components=gmm_params["num_components"],
            reg_lambda=gmm_params["reg_lambda"]
        )
        clients.append(client)
    return clients


def evaluate_global_model(clients, use_test=False):
    """
    全局模型评估：评估每个客户端的个性化模型 (f_i, g_i)，并计算平均性能。
    """
    metrics = []
    for client in clients:
        acc, recall, precision, f1 = client.evaluate(use_test=use_test)
        metrics.append((acc, recall, precision, f1))

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

    # pFedFDA 超参数
    GMM_COMPONENTS = 3  # P_G 和 P_i 的 GMM 分量数
    REG_LAMBDA = 0.01  # 分布正则化损失 L_Reg 的权重
    GEN_LR = 0.0005  # 全局分类器 g_G 的学习率
    GEN_EPOCHS = 3  # 全局分类器 g_G 的训练轮数
    GEN_SAMPLES = 500  # 全局分类器 g_G 训练时的采样数

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

    # 模型参数
    encoder_params = {
        "input_dim": input_dim,
        "hidden_dim": 128,
        "output_dim": 64,  # 特征维度 D
        "num_layers": 3,
        "dropout": 0.4,
    }
    decoder_params = {"hidden_dim": 128, "num_layers": 3, "dropout": 0.3}  # 用于 ResMLP (分类器)

    training_params = {"lr": 0.005, "weight_decay": 1e-4, "local_epochs": 5}
    gmm_params = {"num_components": GMM_COMPONENTS, "reg_lambda": REG_LAMBDA}

    num_rounds = 600
    eval_interval = 1

    # -------- Step 1: 初始化模型 & 服务器 --------
    base_feature_extractor, base_classifier_global, feature_dim = initialize_pFedFDA_models(
        encoder_params, decoder_params, num_classes, device
    )

    # 初始化服务器
    server = PfedfdaServer(
        feature_extractor=base_feature_extractor,
        classifier_global=base_classifier_global,
        device=device,
        feature_dim=feature_dim,
        num_classes=num_classes,
        gmm_components=GMM_COMPONENTS,
        gen_lr=GEN_LR,
        gen_epochs=GEN_EPOCHS
    )

    # 初始化客户端
    clients = load_clients(
        data_paths=pyg_data_files,
        base_feature_extractor=base_feature_extractor,
        base_classifier_global=base_classifier_global,
        training_params=training_params,
        device=device,
        feature_dim=feature_dim,
        gmm_params=gmm_params
    )

    # -------- Step 2: pFedFDA 联邦训练循环 --------
    print("\n================ pFedFDA Federated Training Start ================")
    best_f1 = -1
    best_state = None

    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")

        # 1. Server 分发全局模型 f, g_G 和 P_G 参数
        server.distribute_parameters(clients)

        # 2. Client 本地训练 (f_i, g_i) 并上传更新
        client_updates = []
        print("Client Step: Local training with L_Loc + L_Reg...")
        for client in clients:
            client.local_train(training_params["local_epochs"])
            update = client.get_trained_update()
            client_updates.append(update)
            print(f"Client {client.client_id} finished local training and updated P_i.")

        # 3. Server 聚合与训练
        print("Server Step: Aggregating f and updating P_G/g_G...")

        # a. 聚合特征提取器 f (FedAvg)
        server.aggregate_feature_extractor(client_updates)

        # b. 更新全局特征分布 P_G
        server.update_global_distribution(client_updates)

        # c. 训练全局生成式分类器 g_G (L_Gen)
        server.train_global_classifier(num_samples=GEN_SAMPLES)

        # 4. 评估 (客户端使用本地个性化模型 f_i, g_i 评估)
        if rnd % eval_interval == 0:
            print("\nEvaluation (Validation Set):")
            # 必须在 Server 分发之前进行评估，否则客户端模型可能被重置为全局模型
            avg_acc, avg_recall, avg_prec, avg_f1 = evaluate_global_model(clients, use_test=False)

            # 由于是 pFedFDA，Server 维护的状态是 (f, g_G, P_G)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_state = server.get_global_parameters()
                print("===> New best personalized state saved")

    # -------- Step 3: 最终评估 --------
    print("\n================ pFedFDA Federated Training Finished ================")
    if best_state is not None:
        # 将 Server 恢复到最佳状态，并分发给 Client
        server.set_global_parameters(best_state)
        server.distribute_parameters(clients)

    print("\n================ Final Evaluation (Test Set) ================")
    evaluate_global_model(clients, use_test=True)


if __name__ == "__main__":
    main()