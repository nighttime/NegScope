import math
import pdb
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import *


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, directional=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.directional = directional

        weight_shape = [in_features, out_features]
        if self.directional:
            weight_shape = [2] + weight_shape
        self.weight = Parameter(torch.FloatTensor(*weight_shape))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features, adj):
        # Directional A Matrix
        if self.directional:
            support_in = torch.matmul(features, self.weight[0])
            support_out = torch.matmul(features, self.weight[1])
            # pdb.set_trace()
            output = torch.bmm(adj[:,0],support_in) + torch.bmm(adj[:,1],support_out)

        # Unified A Matrix
        else:
            support = torch.matmul(features, self.weight)
            output = torch.bmm(adj, support)

        # pdb.set_trace()
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class F1_Loss(Module):
    
    def __init__(self):
        super(F1_Loss, self).__init__()
        
    def forward(self, actual, expected):
        actual = actual.double()
        expected = expected.double()

        tp = torch.sum(expected * actual, dim=-1)
        tn = torch.sum((1.0-expected) * (1.0-actual), dim=-1)
        fp = torch.sum((1.0-expected) * actual, dim=-1)
        fn = torch.sum(expected * (1 - actual), dim=-1)

        precision = tp / (tp + fp + EPSILON)
        recall    = tp / (tp + fn + EPSILON)

        F1 = 2 * precision * recall / (precision + recall + EPSILON)
        # F1[torch.isnan(F1)] = 0.
        # pdb.set_trace()
        return 1 - F1.mean()




