import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class CNNDiscriminator2(nn.Module):
    
    def __init__(self, embed_dim, vocab_size, num_classes, filter_sizes,
                 num_filters, padding_idx, dropout=0.2):
        # super(CNNDiscriminator2, self).__init__()
        
        V = vocab_size
        D = embed_dim
        C = num_classes if num_classes > 2 else 1
        Ci = 1
        Co = num_filters
        Ks = filter_sizes

        self.embed = nn.Embedding(V, D, padding_idx=padding_idx)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

        if self.args.static:
            self.embed.weight.requires_grad = False

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        prob = torch.sigmoid(logit)
        return prob

