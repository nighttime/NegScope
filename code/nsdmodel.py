import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import pdb


class GCN(nn.Module):
    def __init__(self, num_layers, input_features, hidden_units, classes, vocab_size):
        assert num_layers >= 2

        super(GCN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, input_features-1)

        self.num_layers = num_layers
        self.layers = \
            [GraphConvolution(input_features, hidden_units)] + \
            [GraphConvolution(hidden_units, hidden_units) for _ in range(num_layers-2)] + \
            [GraphConvolution(hidden_units, classes)]
        # self.dropout = dropout

    def forward(self, x, cue, adj):
        tx = torch.tensor(x)
        ex = self.embedding(tx)
        tc = torch.tensor(cue).float().unsqueeze(1)
        A = torch.FloatTensor(adj)

        x = torch.cat((ex, tc), 1)
        for layer in self.layers:
            x = F.relu(layer(x, A))
            # x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)
