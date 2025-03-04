import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from src.server.fedavg import FedAvgServer
from src.client.fedcp import FedCPClient
from collections import OrderedDict


class Gate(nn.Module):
    def __init__(self, cs: nn.Module):
        super().__init__()
        self.cs = cs
        self.pm = []
        self.gm = []
        self.pm_ = []
        self.gm_ = []

    def forward(self, rep, tau=1.0, hard=False, context=None, flag=0):
        pm, gm = self.cs(context, tau=tau, hard=hard)
        if self.training:
            self.pm.extend(pm)
            self.gm.extend(gm)
        else:
            self.pm_.extend(pm)
            self.gm_.extend(gm)

        if flag == 0:
            rep_p = rep * pm
            rep_g = rep * gm
            return rep_p, rep_g
        elif flag == 1:
            return rep * pm
        else:
            return rep * gm


class ConditionalSelection(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, h_dim * 2),
            nn.LayerNorm([h_dim * 2]),
            nn.ReLU(),
        )

    def forward(self, x, tau=1.0, hard=False):
        shape_ = x.shape
        out = self.fc(x)
        out = out.view(shape_[0], 2, -1)
        out = F.gumbel_softmax(out, dim=1, tau=tau, hard=hard)
        return out[:, 0, :], out[:, 1, :]


class Ensemble(nn.Module):
    def __init__(self, model: nn.Module, cs: nn.Module):
        super().__init__()
        self.base = model.base                              # local 模型
        self.classifier = model.classifier                  # local 模型
        self.model = copy.deepcopy(model)                   # global fixed

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = False

        self.flag = 0
        self.tau = 1
        self.hard = False
        self.context = None

        self.gate = Gate(cs)

    def forward(self, x, is_rep=False, context=None):
        rep_local = self.base(x)

        gate_in = rep_local
        if context is not None:
            context = F.normalize(context, p=2, dim=1)
            batch_size = x[0].shape[0] if isinstance(x, list) else x.shape[0]
            self.context = torch.tile(context, (batch_size, 1))
            gate_in = rep_local * self.context

        if self.flag == 0:  # ghead + phead
            rep_p, rep_g = self.gate(rep_local, self.tau, self.hard, gate_in, flag=0)
            out_local = self.classifier(rep_p)
            out_global = self.model.classifier(rep_g)
            output = out_local + out_global
        elif self.flag == 1:  # phead
            rep_p = self.gate(rep_local, self.tau, self.hard, gate_in, flag=1)
            output = self.classifier(rep_p)
        else:  # ghead
            rep_g = self.gate(rep_local, self.tau, self.hard, gate_in, flag=2)
            output = self.model.classifier(rep_g)

        if is_rep:
            rep_base = self.model.base(x)
            return output, rep_local, rep_base
        else:
            return output

class FedCPServer(FedAvgServer):
    algorithm_name = "fedcp"
    return_diff = False
    client_cls = FedCPClient

    def __init__(self, **commons):
        super().__init__(**commons)

        in_dim = list(self.model.classifier.parameters())[0].shape[1]
        cs = ConditionalSelection(in_dim=in_dim, h_dim=in_dim).to(self.device)
        self.model = Ensemble(model=self.model, cs=cs).to(self.device)

        _init_global_params, _init_global_params_name = [], []
        for key, param in self.model.named_parameters():
            if 'model.classifier.' in key or 'gate' in key or ('model.base.' not in key and 'base.' in key):
                _init_global_params.append(param.data.clone())
                _init_global_params_name.append(key)

        self.public_model_param_names = _init_global_params_name
        self.public_model_params: OrderedDict[str, torch.Tensor] = OrderedDict(
            zip(_init_global_params_name, _init_global_params)
        )

        self.init_trainer()

    def aggregate_client_updates(self, client_packages: Dict[int, Dict[str, Any]]):
        client_weights = [package["weight"] for package in client_packages.values()]
        weights = torch.tensor(client_weights) / sum(client_weights)

        # def receive_models(self) 里面只加了 feature_extractor 对应我们的代码是 base 层参数
        for name in self.public_model_params.keys():
            local_params = []
            for pkg in client_packages.values():
                local_params.append(pkg["regular_model_params"][name].to(self.device))
            local_params = torch.stack(local_params, dim=-1)
            aggregated = torch.sum(local_params * weights, dim=-1)
            self.public_model_params[name].data = aggregated

        self.model.load_state_dict(self.public_model_params, strict=False)