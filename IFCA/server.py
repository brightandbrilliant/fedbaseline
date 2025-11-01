import copy
import torch
from typing import List, Dict, Any


class Server:
    def __init__(self, feature_encoders: List[torch.nn.Module],
                 decoders: List[torch.nn.Module],
                 device: str,
                 num_clusters: int):
        """
        初始化 IFCA 服务器端，管理 K 组集群模型。

        Args:
            feature_encoders (List): K 个集群特征编码器列表。
            decoders (List): K 个集群分类头列表。
            device (str): 计算设备。
            num_clusters (int): 集群总数 K。
        """
        self.num_clusters = num_clusters
        # K 个集群模型
        self.feature_encoders = [enc.to(device) for enc in feature_encoders]
        self.decoders = [dec.to(device) for dec in decoders]

        # 【移除】不再需要结构编码器
        # self.structure_encoders = [...]
        self.device = device

    def get_global_parameters(self) -> Dict[str, Any]:
        """
        获取当前 K 组全局模型的参数。
        """
        return {
            "feature_encoders": [copy.deepcopy(enc.state_dict()) for enc in self.feature_encoders],
            "decoders": [copy.deepcopy(dec.state_dict()) for dec in self.decoders],
            # 【移除】不再返回结构编码器
            # "structure_encoders": [...]
        }

    def set_global_parameters(self, parameters: Dict[str, Any]):
        """
        设置全局模型的参数。
        """
        # 1. 【移除】结构编码器参数加载
        # for idx, enc in enumerate(self.structure_encoders):
        #     enc.load_state_dict(parameters["structure_encoders"][idx])

        # 2. K 个特征编码器
        for idx, enc in enumerate(self.feature_encoders):
            enc.load_state_dict(parameters["feature_encoders"][idx])

        # 3. K 个分类头
        for idx, dec in enumerate(self.decoders):
            dec.load_state_dict(parameters["decoders"][idx])

    def aggregate_all_weights(self, client_updates: List[Dict[str, Any]]):
        """
        根据客户端上传的集群 ID 和模型更新进行聚合。
        """

        # 1. 将客户端更新按集群 ID 分组
        grouped_updates = {j: [] for j in range(self.num_clusters)}
        for update in client_updates:
            j = update["cluster_id"]
            if j < self.num_clusters:
                grouped_updates[j].append(update)

        print("-" * 30)

        # 2. 遍历每个集群，执行 FedAvg 聚合
        for j in range(self.num_clusters):
            updates = grouped_updates[j]

            if not updates:
                print(f"Cluster {j}: No clients participated.")
                continue

            total_samples = sum(update["num_nodes"] for update in updates)
            print(f"Cluster {j}: Aggregating {len(updates)} clients (Total nodes: {total_samples})")

            # --- 2a. 聚合特征编码器 (Feature Encoder) ---
            global_enc_state = self.feature_encoders[j].state_dict()
            new_enc_state = copy.deepcopy(global_enc_state)

            for key in new_enc_state.keys():
                new_enc_state[key] = sum(
                    (update["feature_encoder"][key] * (update["num_nodes"] / total_samples))
                    for update in updates
                )

            self.feature_encoders[j].load_state_dict(new_enc_state)

            # --- 2b. 聚合分类头 (Decoder) ---
            global_dec_state = self.decoders[j].state_dict()
            new_dec_state = copy.deepcopy(global_dec_state)

            for key in new_dec_state.keys():
                new_dec_state[key] = sum(
                    (update["decoder"][key] * (update["num_nodes"] / total_samples))
                    for update in updates
                )

            self.decoders[j].load_state_dict(new_dec_state)

        print("-" * 30)

    def distribute_parameters(self, clients: List[Any]):
        """
        将所有全局参数 (K 组模型) 下发给所有客户端。
        """
        params = self.get_global_parameters()
        for client in clients:
            # client.set_parameters 现只加载 feature_encoders 和 decoders
            client.set_parameters(params)

