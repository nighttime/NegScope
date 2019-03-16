import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import pdb


class GCN(nn.Module):
    def __init__(self, num_layers, input_features, hidden_units, classes, vocab_size):
        assert num_layers >= 2

        super(GCN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, input_features-1)

        self.num_layers = num_layers
        self.layers = nn.ModuleList( \
            [GraphConvolution(input_features, hidden_units)] + \
            [GraphConvolution(hidden_units, hidden_units) for _ in range(num_layers-2)] + \
            [GraphConvolution(hidden_units, classes)])
        # self.dropout = dropout

    def forward(self, x, cue, adj):
        emb = self.embedding(torch.tensor(x))
        tc = torch.tensor(cue).float().unsqueeze(-1)
        A = torch.FloatTensor(adj)
        x = torch.cat((emb, tc), -1)

        # pdb.set_trace()

        for layer in self.layers:
            x = F.relu(layer(x, A))
            # x = F.dropout(x, self.dropout, training=self.training)
        # pdb.set_trace()
        return F.softmax(x, dim=1)
