import copy
import torch
from typing import Any
from collections import OrderedDict

from src.client.fedavg import FedAvgClient


def MMD(x, y, kernel="rbf", device='gpu'):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz

    XX = torch.zeros(xx.shape, device=device)
    YY = torch.zeros(xx.shape, device=device)
    XY = torch.zeros(xx.shape, device=device)

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
    elif kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


class FedCPClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        # lamda 系数 (MMD 强度)
        self.lamda = getattr(self.args, "lamda", 0.0)
        self.regular_params_name = list(key for key, _ in self.model.named_parameters())

    def set_parameters(self, package: dict[str, Any]):
        # def set_parameters(self, feature_extractor) 复现
        super().set_parameters(package)
        self.model.model.base = copy.deepcopy(self.model.base)

        # def set_head_g(self, head) 复现
        with torch.no_grad():
            headw_ps = []
            for name, mat in self.model.classifier.named_parameters():
                if 'weight' in name:
                    headw_ps.append(mat.data)
            if len(headw_ps) > 0:
                headw_p = headw_ps[-1]
                for mat in headw_ps[-2::-1]:
                    headw_p = torch.matmul(headw_p, mat)
                headw_p.detach_()
                self.context = torch.sum(headw_p, dim=0, keepdim=True).to(self.device)

            model_param = package['regular_model_params']

            # Helper function to update parameters
            def update_module_params(module, keyword):
                filtered_params = OrderedDict()
                for key in model_param.keys():
                    if keyword in key:
                        filtered_params[key] = model_param[key]

                for new_param, old_param in zip(filtered_params.values(), module.parameters()):
                    old_param.data = new_param.data.clone()

            # Update classifier and gate parameters using the helper function
            update_module_params(self.model.model.classifier, 'classifier')
            update_module_params(self.model.gate.cs, 'gate')

    def fit(self):
        # client.train_cs_model() 复现
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):

            self.model.gate.pm = []
            self.model.gate.gm = []

            for x, y in self.trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output, rep, rep_base = self.model(x, is_rep=True, context=self.context)
                loss = self.criterion(output, y)
                # 加上 MMD
                loss += MMD(rep, rep_base, kernel='rbf', device=self.device) * self.lamda

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # client.generate_upload_head() 复现
        for (np, pp), (ng, pg) in zip(self.model.classifier.named_parameters(),
                                      self.model.model.classifier.named_parameters()):
            pg.data = pp * 0.5 + pg * 0.5

    def finetune(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.args.common.test.client.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output, rep, rep_base = self.model(x, is_rep=True, context=self.context)
                loss = self.criterion(output, y)
                loss += MMD(rep, rep_base, kernel='rbf', device=self.device) * self.lamda

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()