import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, num_layers, input_features, hidden_units, classes, dropout):
        assert num_layers >= 2

        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.layers = \
            [GraphConvolution(input_features, hidden_units)] + \
            [GraphConvolution(hidden_units, hidden_units) for _ in range(num_layers-2)] + \
            [GraphConvolution(hidden_units, classes)]
        self.dropout = dropout

    def forward(self, x, adj):
        for l in self.layers:
            x = F.relu(self.layers[l](x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)
