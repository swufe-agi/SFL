import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from src.client.fedavg import FedAvgClient


def MMD(x, y, kernel="rbf", device='cpu'):
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


class FedCPClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        # lamda 系数 (MMD 强度)
        self.lamda = getattr(self.args, "lamda", 0.0)

        in_dim = list(self.model.classifier.parameters())[0].shape[1]
        cs = ConditionalSelection(in_dim=in_dim, h_dim=in_dim).to(self.device)
        self.model = Ensemble(model=self.model, cs=cs).to(self.device)

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)

        with torch.no_grad():
            headw_ps = []
            for name, mat in self.model.model.classifier.named_parameters():
                if 'weight' in name:
                    headw_ps.append(mat.data)
            if len(headw_ps) > 0:
                headw_p = headw_ps[-1]
                for mat in headw_ps[-2::-1]:
                    headw_p = torch.matmul(headw_p, mat)
                headw_p.detach_()
                self.context = torch.sum(headw_p, dim=0, keepdim=True).to(self.device)

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):

            self.model.gate.pm = []
            self.model.gate.gm = []

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
                # 加上 MMD
                loss += MMD(rep, rep_base, kernel='rbf', device=self.device) * self.lamda

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

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