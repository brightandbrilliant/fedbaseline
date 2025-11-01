import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import copy


class Client:
    def __init__(self, client_id, data, feature_encoders, decoders,
                 device, lr=1e-4, weight_decay=1e-5, num_clusters=None):
        """
        IFCA 客户端初始化 (纯净版本)。

        Args:
            client_id (...): 客户端ID。
            data (...): 客户端本地数据 (包含节点分类 masks)。
            feature_encoders (list): K 个集群的特征编码器列表。
            decoders (list): K 个集群的分类头列表。
            device (str): 设备。
            num_clusters (int): 集群总数 K。
        """
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.num_clusters = num_clusters
        self.criterion = nn.CrossEntropyLoss()

        # K 个模型列表
        self.feature_encoders = [enc.to(device) for enc in feature_encoders]
        self.decoders = [dec.to(device) for dec in decoders]

        # 存储当前选择的集群索引
        self.current_cluster_id = -1

        self.lr = lr
        self.weight_decay = weight_decay

    def get_optimizer(self, model_index):
        """为指定的集群模型创建优化器。"""
        # 优化参数仅包含 GNN 和分类头
        params = list(self.feature_encoders[model_index].parameters()) + \
                 list(self.decoders[model_index].parameters())

        return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

    def set_parameters(self, parameters):
        """将 K 组全局模型参数下发给客户端"""

        # 1. 【移除】结构编码器参数加载
        # for idx, enc in enumerate(self.structure_encoders):
        #     enc.load_state_dict(parameters["structure_encoders"][idx])

        # 2. K 个特征编码器
        for idx, enc in enumerate(self.feature_encoders):
            enc.load_state_dict(parameters["feature_encoders"][idx])

        # 3. K 个分类头
        for idx, dec in enumerate(self.decoders):
            dec.load_state_dict(parameters["decoders"][idx])

        # 【移除】更新缓存的结构特征
        # self.extract_structure_features()
        # self.z_struct = self.data.structure_x

    def select_best_cluster(self):
        """
        计算本地训练集损失，选择表现最好的模型作为当前集群。
        """
        min_loss = float('inf')
        best_cluster_id = -1

        train_mask = self.data.train_mask
        valid_labels_mask = (self.data.y >= 0)
        final_train_mask = train_mask & valid_labels_mask

        train_labels = self.data.y[final_train_mask]

        if train_labels.numel() == 0:
            best_cluster_id = torch.randint(0, self.num_clusters, (1,)).item()
            self.current_cluster_id = best_cluster_id
            return best_cluster_id

        with torch.no_grad():
            for j in range(self.num_clusters):
                # 1. 设置模型到评估模式
                self.feature_encoders[j].eval()
                self.decoders[j].eval()

                # 2. 前向传播： GNN 输出即为融合特征
                z = self.feature_encoders[j](self.data.x, self.data.edge_index)

                # 【移除】 z = torch.cat([z_feat, self.z_struct], dim=1)

                # 分类头接收 GNN 输出
                logits = self.decoders[j](z)

                # 3. 计算本地训练损失 F_i(theta_j)
                train_logits = logits[final_train_mask]
                loss = self.criterion(train_logits, train_labels)

                if loss.item() < min_loss:
                    min_loss = loss.item()
                    best_cluster_id = j

        self.current_cluster_id = best_cluster_id
        return best_cluster_id

    def local_train(self, epochs):
        if self.current_cluster_id == -1:
            self.select_best_cluster()

        j = self.current_cluster_id

        encoder = self.feature_encoders[j]
        decoder = self.decoders[j]
        optimizer = self.get_optimizer(j)

        encoder.train()
        decoder.train()

        train_mask = self.data.train_mask
        valid_labels_mask = (self.data.y >= 0)
        final_train_mask = train_mask & valid_labels_mask

        if final_train_mask.sum().item() == 0:
            return 0.0

        for _ in range(epochs):
            optimizer.zero_grad()

            # GNN 输出
            z = encoder(self.data.x, self.data.edge_index)

            logits = decoder(z)

            train_logits = logits[final_train_mask]
            train_labels = self.data.y[final_train_mask]

            loss = self.criterion(train_logits, train_labels)

            loss.backward()
            optimizer.step()

        return loss.item()

    def get_trained_update(self):
        """
        返回本地训练后的模型参数和其集群 ID。
        """
        if self.current_cluster_id == -1:
            raise ValueError("Client must select a cluster before getting trained update.")

        j = self.current_cluster_id

        return {
            "cluster_id": j,
            "feature_encoder": copy.deepcopy(self.feature_encoders[j].state_dict()),
            "decoder": copy.deepcopy(self.decoders[j].state_dict()),
            "num_nodes": self.data.num_nodes
        }

    def evaluate(self, use_test=False):
        """
        评估当前客户端模型，使用其当前选择的最佳集群模型。
        """
        j = self.select_best_cluster()

        encoder = self.feature_encoders[j]
        decoder = self.decoders[j]

        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            # GNN 输出
            z = encoder(self.data.x, self.data.edge_index)
            # 【移除】 z = torch.cat([z_feat, self.z_struct], dim=1)

            logits = decoder(z)

            if use_test:
                initial_mask = self.data.test_mask
            else:
                initial_mask = self.data.val_mask

            valid_labels_mask = (self.data.y >= 0)
            final_eval_mask = initial_mask & valid_labels_mask

            if final_eval_mask.sum().item() == 0:
                return 0.0, 0.0, 0.0, 0.0

            eval_logits = logits[final_eval_mask]
            y_true = self.data.y[final_eval_mask]

            y_pred_labels = eval_logits.argmax(dim=1)

            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred_labels.cpu().numpy()

            acc = accuracy_score(y_true_np, y_pred_np)
            recall = recall_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
            precision = precision_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
            f1 = f1_score(y_true_np, y_pred_np, average='weighted', zero_division=0)

            print(f"Client {self.client_id} uses Cluster {j}", end=" | ")

        return acc, recall, precision, f1
