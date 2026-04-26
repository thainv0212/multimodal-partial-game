import torch.nn.functional as func
import torch
import torch.nn as nn
from utils import all_gather_batch_with_grad


class CoMMLoss(nn.Module):
    """
        Normalized Temperature Cross-Entropy Loss for Multi-Modal Contrastive Learning as defined in CoMM [1]

        [1] What to align in multimodal contrastive learning, Dufumier & Castillo-Navarro et al., ICLR 2025
    """

    def __init__(self, temperature=0.1, weights=None):
        super().__init__()
        self.temperature = temperature
        self.weights = weights
        self.INF = 1e8

    def infonce(self, z1, z2):
        N = len(z1)
        sim_zii = (z1 @ z1.T) / self.temperature  # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z2 @ z2.T) / self.temperature  # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z1 @ z2.T) / self.temperature  # dim [N, N] => the diag contains the correct pairs (i,j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        # compute SSL accuracy
        with torch.no_grad():
            pred = torch.argmax(sim_zij, dim=1)
            correct = pred.eq(torch.arange(N, device=z1.device)).sum()
            acc = 100 * correct / N
        return loss, acc

    def forward(self, outputs):
        """
        :param outputs: Dict
            Dictionary with keys:
                - "aug1_embed", List of tensors with shape (bsize, feature_dim), 1st aug.
                - "aug2_embed", List of tensors with shape (bsize, feature_dim), 2nd aug.
                - "prototype", integer indicating where the multimodal representation Z 
                    is stored in "aug1_embed" and "aug2_embed".
        :return: {"loss": torch.Tensor(float), "ssl_acc": torch.Tensor(float)}
        """
        # Prepare embeddings (normalize + gather across all GPU)
        z1, z2, prototype = outputs["aug1_embed"], outputs["aug2_embed"], outputs["prototype"]
        assert len(z1) == len(z2)
        n_emb = len(z1)
        z1 = [func.normalize(z, p=2, dim=-1) for z in z1]
        z2 = [func.normalize(z, p=2, dim=-1) for z in z2]
        Z = all_gather_batch_with_grad(z1 + z2)
        z1, z2 = Z[:n_emb], Z[n_emb:]

        # Apply InfoNCE between a "prototype embedding" and all the others
        loss = []
        acc = []
        for i in range(n_emb):
            loss1, acc1 = self.infonce(z1[i], z2[prototype])
            loss2, acc2 = self.infonce(z2[i], z1[prototype])
            loss.append((loss1 + loss2) / 2.)
            acc.append((acc1 + acc2) / 2.)
        ssl_acc = {"ssl_acc_%i" % i: acc_ for i, acc_ in enumerate(acc)}
        losses = {"ssl_loss_%i" % i: l for i, l in enumerate(loss)}
        if self.weights is not None:
            loss = torch.mean(torch.stack(loss) * torch.tensor(self.weights, device=z1[0].device))
        else:
            loss = torch.mean(torch.stack(loss))
        acc = torch.mean(torch.stack(acc))
        return {"loss": loss, "ssl_acc": acc, **ssl_acc, **losses}

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)


save_num1 = []
save_num2 = []
save_num3 = []
save_num4 = []


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        save_num1.append(x[0].detach().cpu().numpy())
        save_num2.append(x[1].detach().cpu().numpy())
        save_num3.append(self.params[0].detach().clone().cpu().numpy())
        save_num4.append(self.params[1].detach().clone().cpu().numpy())
        # global save_i
        # save_i+=1
        # if(save_i%100==0):
        #     np.save('loss1.npy',np.array(save_num1))
        #     np.save('loss2.npy',np.array(save_num2))
        #     np.save('a1.npy',np.array(save_num3))
        #     np.save('a2.npy',np.array(save_num4))

        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class CoMMWeightedLoss(nn.Module):
    """
        Normalized Temperature Cross-Entropy Loss for Multi-Modal Contrastive Learning as defined in CoMM [1]

        [1] What to align in multimodal contrastive learning, Dufumier & Castillo-Navarro et al., ICLR 2025
    """

    def __init__(self, temperature=0.1, weights=None):
        super().__init__()
        self.temperature = temperature
        self.weights = weights
        self.INF = 1e8
        self.awl = AutomaticWeightedLoss(3)

    def infonce(self, z1, z2):
        N = len(z1)
        sim_zii = (z1 @ z1.T) / self.temperature  # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z2 @ z2.T) / self.temperature  # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z1 @ z2.T) / self.temperature  # dim [N, N] => the diag contains the correct pairs (i,j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        # compute SSL accuracy
        with torch.no_grad():
            pred = torch.argmax(sim_zij, dim=1)
            correct = pred.eq(torch.arange(N, device=z1.device)).sum()
            acc = 100 * correct / N
        return loss, acc

    def forward(self, outputs):
        """
        :param outputs: Dict
            Dictionary with keys:
                - "aug1_embed", List of tensors with shape (bsize, feature_dim), 1st aug.
                - "aug2_embed", List of tensors with shape (bsize, feature_dim), 2nd aug.
                - "prototype", integer indicating where the multimodal representation Z
                    is stored in "aug1_embed" and "aug2_embed".
        :return: {"loss": torch.Tensor(float), "ssl_acc": torch.Tensor(float)}
        """
        # Prepare embeddings (normalize + gather across all GPU)
        z1, z2, prototype = outputs["aug1_embed"], outputs["aug2_embed"], outputs["prototype"]
        assert len(z1) == len(z2)
        n_emb = len(z1)
        z1 = [func.normalize(z, p=2, dim=-1) for z in z1]
        z2 = [func.normalize(z, p=2, dim=-1) for z in z2]
        Z = all_gather_batch_with_grad(z1 + z2)
        z1, z2 = Z[:n_emb], Z[n_emb:]

        # Apply InfoNCE between a "prototype embedding" and all the others
        loss = []
        acc = []
        for i in range(n_emb):
            loss1, acc1 = self.infonce(z1[i], z2[prototype])
            loss2, acc2 = self.infonce(z2[i], z1[prototype])
            loss.append((loss1 + loss2) / 2.)
            # loss.append(self.awl.forward(loss1, loss2))
            acc.append((acc1 + acc2) / 2.)
        ssl_acc = {"ssl_acc_%i" % i: acc_ for i, acc_ in enumerate(acc)}
        losses = {"ssl_loss_%i" % i: l for i, l in enumerate(loss)}
        if self.weights is not None:
            loss = torch.mean(torch.stack(loss) * torch.tensor(self.weights, device=z1[0].device))
        else:
            # loss = torch.mean(torch.stack(loss))
            loss = self.awl.forward(*loss)
        acc = torch.mean(torch.stack(acc))
        return {"loss": loss, "ssl_acc": acc, **ssl_acc, **losses}

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)


class CoMMWeightedLoss2(nn.Module):
    """
        Normalized Temperature Cross-Entropy Loss for Multi-Modal Contrastive Learning as defined in CoMM [1]

        [1] What to align in multimodal contrastive learning, Dufumier & Castillo-Navarro et al., ICLR 2025
    """

    def __init__(self, temperature=0.1, weights=None):
        super().__init__()
        self.temperature = temperature
        self.weights = weights
        self.INF = 1e8
        self.awl = AutomaticWeightedLoss(2)

    def infonce(self, z1, z2):
        N = len(z1)
        sim_zii = (z1 @ z1.T) / self.temperature  # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z2 @ z2.T) / self.temperature  # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z1 @ z2.T) / self.temperature  # dim [N, N] => the diag contains the correct pairs (i,j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        # compute SSL accuracy
        with torch.no_grad():
            pred = torch.argmax(sim_zij, dim=1)
            correct = pred.eq(torch.arange(N, device=z1.device)).sum()
            acc = 100 * correct / N
        return loss, acc

    def forward(self, outputs):
        """
        :param outputs: Dict
            Dictionary with keys:
                - "aug1_embed", List of tensors with shape (bsize, feature_dim), 1st aug.
                - "aug2_embed", List of tensors with shape (bsize, feature_dim), 2nd aug.
                - "prototype", integer indicating where the multimodal representation Z
                    is stored in "aug1_embed" and "aug2_embed".
        :return: {"loss": torch.Tensor(float), "ssl_acc": torch.Tensor(float)}
        """
        # Prepare embeddings (normalize + gather across all GPU)
        z1, z2, prototype = outputs["aug1_embed"], outputs["aug2_embed"], outputs["prototype"]
        assert len(z1) == len(z2)
        n_emb = len(z1)
        z1 = [func.normalize(z, p=2, dim=-1) for z in z1]
        z2 = [func.normalize(z, p=2, dim=-1) for z in z2]
        Z = all_gather_batch_with_grad(z1 + z2)
        z1, z2 = Z[:n_emb], Z[n_emb:]

        # Apply InfoNCE between a "prototype embedding" and all the others
        loss = []
        acc = []
        for i in range(n_emb - 1):
            loss1, acc1 = self.infonce(z1[i], z2[prototype])
            loss2, acc2 = self.infonce(z2[i], z1[prototype])
            loss.append((loss1 + loss2) / 2.)
            # loss.append(self.awl.forward(loss1, loss2))
            acc.append((acc1 + acc2) / 2.)
        ssl_acc = {"ssl_acc_%i" % i: acc_ for i, acc_ in enumerate(acc)}
        losses = {"ssl_loss_%i" % i: l for i, l in enumerate(loss)}
        if self.weights is not None:
            loss = torch.mean(torch.stack(loss) * torch.tensor(self.weights, device=z1[0].device))
        else:
            loss = torch.mean(torch.stack(loss))
            # loss = self.awl.forward(*loss)

        # 1st loss component
        loss_1st = 0
        i = n_emb - 1
        loss1, acc1 = self.infonce(z1[i], z2[prototype])
        loss2, acc2 = self.infonce(z2[i], z1[prototype])
        loss_1st = (loss1 + loss2) / 2.
        acc.append((acc1 + acc2) / 2.)
        ssl_acc["ssl_acc_%i" % i] = (acc1 + acc2) / 2.
        losses["ssl_loss_%i" % i] = loss_1st
        # weighted loss
        loss = self.awl.forward(loss_1st, loss)
        acc = torch.mean(torch.stack(acc))
        return {"loss": loss, "ssl_acc": acc, **ssl_acc, **losses}

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)