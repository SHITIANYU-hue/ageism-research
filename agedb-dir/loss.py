########################################################################################
# Code is based on the LDS and FDS (https://arxiv.org/pdf/2102.09554.pdf) implementation
# from https://github.com/YyzHarry/imbalanced-regression/tree/main/imdb-wiki-dir 
# by Yuzhe Yang et al.
########################################################################################
import torch
import torch.nn.functional as F


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

class SupCRLoss:
    def __init__(self, t=1, device='cuda:0'):
        self.t = torch.tensor(t, device=device)  # Create a tensor for self.t
        self.device = device

    def supcr(self, batch, labels):
        batch = batch.to(self.device).double()
        N = len(batch)

        # Reshape labels to remove the extra dimension
        labels = labels.squeeze().to(self.device)

        dists = torch.cdist(labels.unsqueeze(1), labels.unsqueeze(1), p=1).to(self.device)

        I = torch.eye(N).to(self.device)
        zI = (torch.ones((N, N)).to(self.device) - I)

        sims = torch.cdist(batch, batch).to(self.device) * (-1)
        sims = torch.exp((sims * zI) / self.t) - I

        past_thresh = torch.zeros((N, N, N)).to(self.device)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                past_thresh[i][j] = (dists[i] >= dists[i][j])

        past_thresh = past_thresh.permute(0, 2, 1)
        sum3 = (sims.unsqueeze(2) * past_thresh).sum(1) + I

        out = torch.log(sims / sum3 + I) * (-1 / (N * (N - 1)))

        return out



def weighted_supcr_loss(inputs, targets, weights=None, t=1, device='cuda:0'):
    supcr_loss = SupCRLoss(t=t, device=device)
    
    # Reshape the labels tensor
    targets = targets.squeeze()

    loss = supcr_loss.supcr(inputs, targets)
    if weights is not None:
        loss *= weights.expand_as(loss)
    
    loss = torch.mean(loss)
    return loss



def weighted_focal_mse_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, beta=1., weights=None):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
