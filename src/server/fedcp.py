import torch
from typing import Any, Dict
from src.server.fedavg import FedAvgServer
from src.client.fedcp import FedCPClient


class FedCPServer(FedAvgServer):
    """
    继承自原始 FedAvgServer,
    实现 FedCP 的聚合思路 (聚合 feature_extractor)
    """
    algorithm_name = "fedcp"
    return_diff = False
    client_cls = FedCPClient

    def __init__(self, **commons):
        super().__init__(**commons)

    def aggregate_client_updates(self, client_packages: Dict[int, Dict[str, Any]]):
        """
        结合 FedCP 的思路，将上传的多路参数(feature_extractor) 聚合。
          - 只对 local feature_extractor 做 FedAvg
        """
        # 先计算各客户端的权重
        client_weights = [package["weight"] for package in client_packages.values()]
        weights = torch.tensor(client_weights) / sum(client_weights)

        # ========== (A) 聚合 feature_extractor ==========
        # 这里假设 feature_extractor 在 regular_model_params 中
        # 并且 param_name 带有 "feature_extractor."
        # 你可以根据实际命名规则筛选
        for name in self.public_model_params.keys():
            # 判断是否是 feature_extractor
            if "base" in name:
                # 如果是差分方式
                if self.return_diff and "model_params_diff" in client_packages[list(client_packages.keys())[0]]:
                    # 累加 diff
                    diffs = []
                    for pkg in client_packages.values():
                        # pkg["model_params_diff"] 里存的是 diff = global - local
                        # FedCP 可能不一定用 diff，但这里保留写法以防万一
                        diffs.append(pkg["model_params_diff"][name].to(self.device))
                    diffs = torch.stack(diffs, dim=-1)
                    aggregated = torch.sum(diffs * weights, dim=-1)
                    self.public_model_params[name].data -= aggregated
                else:
                    # 常规 FedAvg
                    local_params = []
                    for pkg in client_packages.values():
                        local_params.append(pkg["regular_model_params"][name].to(self.device))
                    local_params = torch.stack(local_params, dim=-1)
                    aggregated = torch.sum(local_params * weights, dim=-1)
                    self.public_model_params[name].data = aggregated

        # ========== (E) 将聚合后的全局参数更新到 self.model 中，以便后续测试/下发给客户端 ==========
        self.model.load_state_dict(self.public_model_params, strict=False)