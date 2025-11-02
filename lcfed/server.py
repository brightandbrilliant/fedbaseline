import copy
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict


class LcServer:
    """
    LCFed 服务器实现。管理全局模型 (theta_Global)，并执行偏差聚类和聚合。
    """

    def __init__(self, global_model, device, num_clusters: int):
        """
        初始化 LCFed 服务器端。

        Args:
            global_model (nn.Module): 全局模型 theta_Global。
            device (torch.device): 计算设备。
            num_clusters (int): 集群数量 K。
        """
        self.device = device
        self.num_clusters = num_clusters

        # 全局共享模型 theta_Global
        self.global_model = global_model.to(device)

        # 存储 K 个集群的偏差： {cluster_id: Delta_theta_Cluster_k}
        self.cluster_biases: Dict[int, Dict[str, torch.Tensor]] = {
            k: self._get_zero_bias() for k in range(num_clusters)
        }

        # 存储客户端到集群 ID 的映射
        self.client_cluster_map: Dict[int, int] = {}

        # 客户端总数 (用于评估等)
        self.num_total_clients = 0

        # --------------------------------------------------------------------------

    # 辅助函数：创建全零的偏差张量 (与 global_model 结构一致)
    # --------------------------------------------------------------------------
    def _get_zero_bias(self) -> Dict[str, torch.Tensor]:
        """返回一个与全局模型结构相同的全零参数字典，用作初始集群偏差。"""
        zero_bias = {}
        for key, param in self.global_model.named_parameters():
            # 使用 torch.zeros_like 确保维度正确，且在 GPU 上
            zero_bias[key] = torch.zeros_like(param.data, device=self.device)
        return zero_bias

    # --------------------------------------------------------------------------
    # 获取/设置全局参数
    # --------------------------------------------------------------------------
    def get_global_parameters(self) -> Dict[str, Any]:
        """
        获取当前全局模型 theta_Global 和所有集群偏差 Delta_theta_Cluster, k 的参数。
        """
        # 全局模型
        global_state = copy.deepcopy(self.global_model.state_dict())

        # K 个集群偏差
        cluster_biases_copy = {
            k: copy.deepcopy(bias) for k, bias in self.cluster_biases.items()
        }

        return {
            "global_model": global_state,
            "cluster_biases": cluster_biases_copy,
            "client_cluster_map": self.client_cluster_map
        }

    def set_global_parameters(self, parameters: Dict[str, Any]):
        """
        设置全局模型的参数。
        """
        self.global_model.load_state_dict(parameters["global_model"])
        self.cluster_biases = parameters["cluster_biases"]
        self.client_cluster_map = parameters["client_cluster_map"]

    # --------------------------------------------------------------------------
    # 辅助函数：将模型参数扁平化为 NumPy 向量
    # --------------------------------------------------------------------------
    def _flatten_bias(self, bias_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """将一个偏差字典扁平化为一个 NumPy 向量。"""
        flat_list = [v.cpu().numpy().flatten() for v in bias_dict.values()]
        return np.concatenate(flat_list)

    # --------------------------------------------------------------------------
    # 辅助函数：将 NumPy 向量还原为模型参数字典
    # --------------------------------------------------------------------------
    def _unflatten_bias(self, flat_vector: np.ndarray, template_bias: Dict[str, torch.Tensor]) -> Dict[
        str, torch.Tensor]:
        """将一个 NumPy 向量还原为模型偏差字典。"""
        new_bias = {}
        cursor = 0
        for key, template_tensor in template_bias.items():
            param_size = template_tensor.numel()
            # 从扁平向量中切出对应的部分，并重塑为原始形状
            new_tensor = torch.from_numpy(flat_vector[cursor:cursor + param_size]).float()
            new_tensor = new_tensor.reshape(template_tensor.shape).to(self.device)
            new_bias[key] = new_tensor
            cursor += param_size
        return new_bias

    # --------------------------------------------------------------------------
    # 【核心 4】执行 K-means 聚类并更新集群映射
    # --------------------------------------------------------------------------
    def cluster_personalized_bias(self, client_updates: List[Dict[str, Any]], client_ids: List[int]):
        """
        对客户端上传的个性化偏差 Delta_theta_i 进行 K-means 聚类。
        """
        # 1. 收集所有偏差向量
        all_bias_vectors = []
        # 使用第一个偏差字典作为模板，用于后续的扁平化/反扁平化
        bias_template = client_updates[0]["personalized_bias"]

        for update in client_updates:
            flat_bias = self._flatten_bias(update["personalized_bias"])
            all_bias_vectors.append(flat_bias)

        X = np.stack(all_bias_vectors)

        # 2. 执行 K-means 聚类
        print(f"Server: Clustering {X.shape[0]} client biases into {self.num_clusters} clusters...")

        # 确保数据量大于集群数
        k_to_use = min(self.num_clusters, X.shape[0])

        # 注意: 这里可能会触发 sklearn 的 MKL 警告
        kmeans = KMeans(n_clusters=k_to_use, random_state=0, n_init='auto').fit(X)

        # 3. 更新客户端到集群 ID 的映射
        self.client_cluster_map = {}
        for i, client_id in enumerate(client_ids):
            cluster_id = int(kmeans.labels_[i])
            self.client_cluster_map[client_id] = cluster_id

        print(f"Clustering finished. Mapped clients: {list(self.client_cluster_map.items())}")

    # --------------------------------------------------------------------------
    # 【核心 5】聚合模型：全局 FedAvg (theta_Global)
    # --------------------------------------------------------------------------
    def aggregate_global_model(self, client_updates: List[Dict[str, Any]]):
        """
        对客户端上传的 theta_i 进行 FedAvg 聚合，更新 theta_Global。
        """
        if not client_updates:
            return

        total_samples = sum(update["num_nodes"] for update in client_updates)

        global_state = self.global_model.state_dict()
        new_state = copy.deepcopy(global_state)

        for key in new_state.keys():
            # 计算加权平均： sum(w_i * state_i)
            new_state[key] = sum(
                (update["local_model"][key] * (update["num_nodes"] / total_samples))
                for update in client_updates
            )

        self.global_model.load_state_dict(new_state)
        print("Global model theta_Global updated via FedAvg.")

    # --------------------------------------------------------------------------
    # 【核心 6】聚合偏差：集群 FedAvg (Delta_theta_Cluster, k)
    # --------------------------------------------------------------------------
    def aggregate_cluster_bias(self, client_updates: List[Dict[str, Any]], client_ids: List[int]):
        """
        根据客户端的集群 ID，在集群内对 Delta_theta_i 进行加权聚合。
        """

        # 1. 按集群 ID 分组客户端更新和节点数
        grouped_updates = defaultdict(list)
        grouped_nodes = defaultdict(int)

        for update, client_id in zip(client_updates, client_ids):
            cluster_id = self.client_cluster_map.get(client_id, -1)
            if cluster_id != -1:
                grouped_updates[cluster_id].append(update["personalized_bias"])
                grouped_nodes[cluster_id] += update["num_nodes"]

        # 2. 遍历每个集群，执行偏差 FedAvg
        for k in range(self.num_clusters):
            biases_k = grouped_updates[k]
            total_nodes_k = grouped_nodes[k]

            if not biases_k:
                print(f"Cluster {k}: No updates received. Bias remains unchanged.")
                continue

            # 使用第一个偏差作为模板（必须保证所有模型的参数结构相同）
            template_bias = biases_k[0]
            new_bias = self._get_zero_bias()

            print(f"Cluster {k}: Aggregating {len(biases_k)} personalized biases.")

            # 遍历模型参数的键
            for key in template_bias.keys():

                # 计算加权平均： sum_{i in C_k} (N_i / N_total_k) * Delta_theta_i

                # 重新计算权重 N_i / N_total_k
                key_sum = torch.zeros_like(new_bias[key], device=self.device)
                current_sum_nodes = 0

                # 找到属于该集群的客户端 N_i
                for update, client_id in zip(client_updates, client_ids):
                    if self.client_cluster_map.get(client_id) == k:
                        current_sum_nodes += update["num_nodes"]

                # 执行加权求和
                if current_sum_nodes > 0:
                    for update, client_id in zip(client_updates, client_ids):
                        if self.client_cluster_map.get(client_id) == k:
                            weight = update["num_nodes"] / current_sum_nodes
                            key_sum += update["personalized_bias"][key] * weight

                new_bias[key] = key_sum

            # 更新集群偏差
            self.cluster_biases[k] = new_bias

    # --------------------------------------------------------------------------
    # 分发：将全局模型和对应集群偏差分发给客户端
    # --------------------------------------------------------------------------
    def distribute_parameters(self, clients: List[Any]):
        """
        将全局模型 theta_Global 和本客户端所属集群的偏差 Delta_theta_Cluster, k 分发给所有客户端。
        """
        # 全局模型参数
        global_state = self.global_model.state_dict()

        for client in clients:
            client_id = client.client_id

            # 1. 获取客户端当前的集群 ID (来自上一轮的聚类结果)
            # 如果是第一轮，或者聚类失败，默认为集群 0
            cluster_id = self.client_cluster_map.get(client_id, 0)

            # 2. 获取该集群的偏差
            # 如果集群 ID 超出范围或不存在，使用全零偏差
            cluster_bias = self.cluster_biases.get(cluster_id, self._get_zero_bias())

            # 3. 构造并下发参数
            params_to_client = {
                "global_model": copy.deepcopy(global_state),
                "cluster_bias": cluster_bias,
                "cluster_id": cluster_id  # 将其集群 ID 告知客户端 (用于评估时的信息输出)
            }

            client.set_parameters(params_to_client)
            