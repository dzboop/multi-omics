import torch
import torch.nn as nn
import math


class Loss(nn.Module):
    def __init__(self, batch_size, temperature_f, temperature_c, temperature_l):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.temperature_c = temperature_c
        self.temperature_l = temperature_l

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_label(self, q_i, q_j):
        q_i = q_i.t()
        q_j = q_j.t()
        class_num = q_i.shape[0]
        N = 2 * class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_c
        sim_i_j = torch.diag(sim, class_num)
        sim_j_i = torch.diag(sim, -class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def contrastive_loss(self, view1_feature, view2_feature, t=0.21):
        #view1_feature = F.normalize(view1_feature, dim=1)
        #view2_feature = F.normalize(view2_feature, dim=1)
        # cosine similarity: NxN
        sim_view12 = self.similarity(view1_feature.unsqueeze(1), view2_feature.unsqueeze(0)) / self.temperature_l

        # logits: NxN
        logits_view12 = sim_view12 - torch.log(torch.exp(1.06 * sim_view12).sum(1, keepdim=True))
        logits_view21 = sim_view12.T - torch.log(torch.exp(1.06 * sim_view12.T).sum(1, keepdim=True))

        # unsupervised cross-view contrastive loss
        loss = - torch.diag(logits_view12).mean() - torch.diag(logits_view21).mean()

        return loss
