import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import pdb


class GCN(nn.Module):
    def __init__(self, num_layers, input_features, hidden_units, classes, vocab_size, directional=False):
        assert num_layers >= 1

        super(GCN, self).__init__()

        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, input_features-1)
        self.convert_to_hidden = nn.Linear(input_features, hidden_units)
        self.graph_conv = GraphConvolution(hidden_units, hidden_units, directional=directional)
        
        # self.layers = nn.ModuleList( \
        #     [GraphConvolution(input_features, hidden_units, directional=directional)] + \
        #     [GraphConvolution(hidden_units, hidden_units, directional=directional) for _ in range(num_layers-1)])
        # self.layer_conversion = GraphConvolution(input_features, hidden_units, directional=directional)
        
        # self.dropout = dropout

        self.classifier = nn.Linear(hidden_units, classes)

    def forward(self, x, cue, adj):
        emb = self.embedding(torch.tensor(x))
        tc = torch.tensor(cue).float().unsqueeze(-1)
        A = torch.FloatTensor(adj)
        x = torch.cat((emb, tc), -1)

        # pdb.set_trace()

        x = F.relu(self.convert_to_hidden(x))
        for _ in range(self.num_layers):
            x = F.relu(self.graph_conv(x, A))

        # for layer in self.layers:
        #     x = F.relu(layer(x, A))
            # x = F.dropout(x, self.dropout, training=self.training)

        output = self.classifier(x)
        # pdb.set_trace()
        return F.softmax(output, dim=-1)
