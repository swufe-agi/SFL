import torch
from typing import Dict, Any
from src.server.fedavg import FedAvgServer
from src.client.fedcp import FedCPClient


class FedCPServer(FedAvgServer):
    algorithm_name = "fedcp"
    return_diff = False
    client_cls = FedCPClient

    def __init__(self, **commons):
        super().__init__(**commons)

    def aggregate_client_updates(self, client_packages: Dict[int, Dict[str, Any]]):
        client_weights = [package["weight"] for package in client_packages.values()]
        weights = torch.tensor(client_weights) / sum(client_weights)

        for name in self.public_model_params.keys():
            if "base" in name:
                if self.return_diff and "model_params_diff" in client_packages[list(client_packages.keys())[0]]:
                    diffs = []
                    for pkg in client_packages.values():
                        diffs.append(pkg["model_params_diff"][name].to(self.device))
                    diffs = torch.stack(diffs, dim=-1)
                    aggregated = torch.sum(diffs * weights, dim=-1)
                    self.public_model_params[name].data -= aggregated
                else:
                    local_params = []
                    for pkg in client_packages.values():
                        local_params.append(pkg["regular_model_params"][name].to(self.device))
                    local_params = torch.stack(local_params, dim=-1)
                    aggregated = torch.sum(local_params * weights, dim=-1)
                    self.public_model_params[name].data = aggregated

        self.model.load_state_dict(self.public_model_params, strict=False)