import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Any

from omegaconf import DictConfig
from src.client.fedavg import FedAvgClient


# =========== 以下是 clientCP 中用到的工具函数 ===========
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


class ConditionalSelection(nn.Module):
    """
    同 FedCP 中的门控网络。示例仅保留 forward 关键逻辑。
    """
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, h_dim * 2),
            nn.LayerNorm([h_dim * 2]),
            nn.ReLU(),
        )

    def forward(self, x, tau=1.0, hard=False):
        """
        x 的 shape: [batch_size, in_dim]
        返回 pm, gm 两个概率 mask
        """
        shape_ = x.shape
        out = self.fc(x)
        out = out.view(shape_[0], 2, -1)  # 分成 2个通道
        out = F.gumbel_softmax(out, dim=1, tau=tau, hard=hard)
        return out[:, 0, :], out[:, 1, :]


class Gate(nn.Module):
    def __init__(self, cs: nn.Module):
        """
        cs: ConditionalSelection子网络 (clientCP 传入)
        """
        super().__init__()
        self.cs = cs
        self.pm = []
        self.gm = []
        self.pm_ = []
        self.gm_ = []

    def forward(self, rep, tau=1.0, hard=False, context=None, flag=0):
        pm, gm = self.cs(context, tau=tau, hard=hard)
        # 训练/推理时分别存储
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


class Ensemble(nn.Module):
    """
    组合本地模型 (含 local classifier + local base) 与全局 head (head_g)
    并用 Gate 模块拆分特征
    """
    def __init__(self, model: nn.Module, cs: nn.Module,
                 classifier: nn.Module, base: nn.Module):
        super().__init__()
        self.model = model                               # 全局固定的 model
        self.classifier = classifier                     # 本地训练 classifier
        self.base = base                                 # 本地训练 base

        for param in self.classifier.parameters():
            param.requires_grad = False
        for param in self.base.parameters():
            param.requires_grad = False

        self.flag = 0
        self.tau = 1
        self.hard = False
        self.context = None

        self.gate = Gate(cs)

    def forward(self, x, is_rep=False, context=None):
        # 从本地可训练分支获取特征
        rep_local = self.base(x)

        gate_in = rep_local
        if context is not None:
            context = F.normalize(context, p=2, dim=1)
            batch_size = x[0].shape[0] if isinstance(x, list) else x.shape[0]
            self.context = torch.tile(context, (batch_size, 1))
            gate_in = rep_local * self.context

        if self.flag == 0:
            rep_p, rep_g = self.gate(rep_local, self.tau, self.hard, gate_in, flag=0)
            out_local = self.classifier(rep_p)
            out_global = self.model.classifier(rep_g)
            output = out_local + out_global
        elif self.flag == 1:
            rep_p = self.gate(rep_local, self.tau, self.hard, gate_in, flag=1)
            output = self.classifier(rep_p)
        else:
            rep_g = self.gate(rep_local, self.tau, self.hard, gate_in, flag=2)
            output = self.model.classifier(rep_g)

        # 如果需要返回特征, 还要给出一个固定的全局特征
        if is_rep:
            rep_base = self.base(x)
            return output, rep_local, rep_base
        else:
            return output


# =============== 核心：FedAvgCPClient 继承自 FedAvgClient ===============
class FedCPClient(FedAvgClient):
    """
    只改写必要的部分，将 clientCP 的逻辑 (MMD + Ensemble + Gate + context) 融入。
    """
    def __init__(
        self,
        model,
        optimizer_cls: type[torch.optim.Optimizer],
        lr_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler],
        args: DictConfig,
        dataset,
        data_indices: list,
        device: torch.device | None,
        return_diff: bool,
    ):
        # 先初始化父类，保证基础联邦逻辑成立
        super().__init__(
            model=model,
            optimizer_cls=optimizer_cls,
            lr_scheduler_cls=lr_scheduler_cls,
            args=args,
            dataset=dataset,
            data_indices=data_indices,
            device=device,
            return_diff=return_diff,
        )

        # =========== 以下整合 clientCP 的部分 =============
        # lamda 系数 (MMD 强度)
        self.lamda = getattr(args, "lamda", 0.0)

        in_dim = list(self.model.classifier.parameters())[0].shape[1]
        cs = ConditionalSelection(in_dim=in_dim, h_dim=in_dim).to(self.device)

        # 将父类中的 self.model 替换为 Ensemble 封装
        # 假设 self.model 中包含 self.model.head / self.model.base
        # 如果结构不同，需要自行改动
        self.model = Ensemble(
            model=copy.deepcopy(self.model),
            cs=cs,
            classifier=copy.deepcopy(self.model.classifier),
            base=copy.deepcopy(self.model.base)
        ).to(self.device)

    def set_parameters(self, package: dict[str, Any]):
        """
        覆盖父类的 set_parameters, 先调用 super().set_parameters()
        再额外生成 context (模仿 clientCP.set_head_g 的逻辑)。
        """
        super().set_parameters(package)

        # 参考 clientCP.set_head_g, 生成 self.context
        # 这里仅示例与 clientCP 相同做法：
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

    def fit(self):
        """
        覆盖父类的本地训练逻辑，在计算 loss 时加上 MMD(rep, rep_base).
        其余不变。
        """
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            # 每个 epoch 前可清空门控统计(可选)
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

                # 与 clientCP.train_cs_model 一致: 拿到3路 (output, rep, rep_base)
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
        """
        测试阶段如果需要微调，再加上 MMD 约束
        """
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