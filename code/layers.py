import math
import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from utils import *

class GCN(Module):
    def __init__(self, num_layers, input_features, hidden_units, classes, vocab_size, directional=False):
        assert num_layers >= 1

        super(GCN, self).__init__()

        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, input_features-1)
        self.convert_to_hidden = nn.Linear(input_features, hidden_units)
        self.graph_conv = GraphConvolution(hidden_units, directional=directional)
        
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
            x = torch.sigmoid(self.graph_conv(x, A))

        # for layer in self.layers:
        #     x = F.relu(layer(x, A))
            # x = F.dropout(x, self.dropout, training=self.training)

        output = self.classifier(x)
        # pdb.set_trace()
        return F.softmax(output, dim=-1)

class GraphConvolution(Module):
    
    def __init__(self, feature_size, bias=True, directional=False, attention=True):
        super(GraphConvolution, self).__init__()
        self.in_features = feature_size
        self.out_features = feature_size
        self.feature_size = feature_size
        # self.directional = directional

        weight_shape = [self.in_features, self.out_features]
        # if self.directional:
        #     weight_shape = [2] + weight_shape
        self.weight = Parameter(torch.FloatTensor(*weight_shape))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.attention = attention
        self.atn_feature_size = 64
        self.atn_q = nn.Linear(feature_size, self.atn_feature_size, bias=False)
        self.atn_k = nn.Linear(feature_size, self.atn_feature_size, bias=False)
        self.atn_v = nn.Linear(feature_size, feature_size, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features, adj):
        # Directional A Matrix
        # if self.directional:
        #     support_in = torch.matmul(features, self.weight[0])
        #     support_out = torch.matmul(features, self.weight[1])
        #     output = torch.bmm(adj[:,0],support_in) + torch.bmm(adj[:,1],support_out)

        # Unified A Matrix
        # else:
        if self.attention:
            bsize, adjsize = adj.shape[0], adj.shape[-1]
            output = torch.zeros(bsize, adjsize, self.feature_size)
            for b in range(bsize):
                # Extract a neighbor feature matrix for each item in the batch
                connections = adj[b]

                # Iterate over nodes in the graph
                for i in range(adjsize):
                    cx = connections[i]
                    if torch.sum(cx) == 0:
                        continue
                    neighbor_feats = features[b,cx.byte(),:]

                    # Compute a query vector for the current node
                    query = self.atn_q(features[b,i]).squeeze()
                    # Compute keys for the neighbor nodes
                    keys = self.atn_k(neighbor_feats)
                    # Compute values for the neighbor nodes
                    values = self.atn_v(neighbor_feats)
                    
                    # Convolve neighbors using the score weightings
                    scores = F.softmax(torch.matmul(keys, query), dim=0)

                    output[b,i] = torch.matmul(scores, values)
            # pdb.set_trace()
        else:
            support = torch.matmul(features, self.weight)
            output = torch.bmm(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# class GraphConvolution(Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """

#     def __init__(self, in_features, out_features, bias=True, directional=False):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.directional = directional

#         weight_shape = [in_features, out_features]
#         if self.directional:
#             weight_shape = [2] + weight_shape
#         self.weight = Parameter(torch.FloatTensor(*weight_shape))
        
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
        
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(-1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, features, adj):
#         # Directional A Matrix
#         if self.directional:
#             support_in = torch.matmul(features, self.weight[0])
#             support_out = torch.matmul(features, self.weight[1])
#             # pdb.set_trace()
#             output = torch.bmm(adj[:,0],support_in) + torch.bmm(adj[:,1],support_out)

#         # Unified A Matrix
#         else:
#             support = torch.matmul(features, self.weight)
#             output = torch.bmm(adj, support)
        
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'

class RecurrentTagger(Module):
    def __init__(self, in_size, out_size, classes, vocab_size, pos_size, pos_vocab_size):
        super(RecurrentTagger, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.word_embedding = nn.Embedding(vocab_size, in_size-1)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_size)
        # self.lstm = nn.LSTM(in_size, out_size)
        self.bilstm = nn.LSTM(in_size+pos_size, out_size, bidirectional=True)
        self.classifier = nn.Linear(out_size*2, classes)

    def forward(self, x, pos, cue):
        word_emb = self.word_embedding(torch.tensor(x))
        pos_emb  = self.pos_embedding(torch.tensor(pos))
        cue_emb  = torch.tensor(cue).float().unsqueeze(-1)

        lstm_input = torch.cat((torch.cat((word_emb, cue_emb), -1), pos_emb), -1)
        
        # lstm_output, _ = self.lstm(lstm_input)
        lstm_output2, _ = self.bilstm(lstm_input)

        class_scores = torch.sigmoid(lstm_output2)
        class_scores = self.classifier(class_scores)

        yhat = F.softmax(class_scores, dim=-1)

        return yhat


class TreeRecurrentTagger(Module):
    def __init__(self, hidden_size, num_classes, vocab_size):
        super(TreeRecurrentTagger, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.word_embedding = nn.Embedding(vocab_size, self.hidden_size-self.num_classes)
        self.compose = nn.Linear(2*hidden_size, hidden_size)
        self.compose2 = nn.Linear(hidden_size, hidden_size)
        self.decompose = nn.Linear(2*hidden_size, 2*hidden_size)
        self.decompose2 = nn.Linear(2*hidden_size, 2*hidden_size)
        self.global_reverse = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x, word_index, cue, adj):
        assert len(adj.shape) == 4
        bsize = x.shape[0]
        dsize = x.shape[1]
        results = torch.zeros(bsize, dsize, self.num_classes)
        
        word_emb = self.word_embedding(torch.tensor(x))
        cue_emb  = make_one_hot(cue, self.num_classes)
        # cue_emb  = torch.tensor(cue).float().unsqueeze(-1)
        in_feats = torch.cat((word_emb, cue_emb), -1)

        # for each sentence in batch
        for i in range(bsize):
            # word_emb = self.word_embedding(torch.tensor(x))
            w_idx = word_index[i]
            sent = in_feats[i,w_idx]
            A = adj[i,1]

            # traverse up the tree
            upward_results = torch.zeros(A.shape[-1], self.hidden_size)
            global_up = self.rec_traverse_up(sent, w_idx, A, 0, upward_results)
            global_down = torch.sigmoid(self.global_reverse(global_up))
            # traverse back down the tree
            downward_results = torch.zeros(dsize, self.hidden_size)
            self.rec_traverse_down(w_idx, A, 0, global_down, downward_results, upward_results)

            # TODO concatenate the upward and downward results before classification
            tree_data = torch.cat((downward_results, upward_results), dim=-1)
            classifications = self.classifier(tree_data)
            class_scores = torch.sigmoid(classifications)

            yhat = F.softmax(class_scores, dim=-1)

            results[i] = yhat
            # pdb.set_trace()

        return results

    def rec_traverse_up(self, x_emb, word_index, A_p2c, parent_idx, result_table):
        def get_child(idx):
            if idx in word_index:
                locs = np.where(word_index==idx)[0]
                assert len(locs) == 1
                return x_emb[locs[0]]
            else:
                return self.rec_traverse_up(x_emb, word_index, A_p2c, idx, result_table)

        children_idx = np.where(A_p2c[parent_idx] == 1)[0]
        assert len(children_idx) in (0, 2)

        left  = get_child(children_idx[0])
        right = get_child(children_idx[1])

        parent_context = torch.cat((left, right), dim=-1)
        parent_rep = torch.tanh(self.compose(parent_context))
        # parent_rep  = torch.sigmoid(self.compose2(parent_rep1))

        result_table[parent_idx] = parent_rep

        return parent_rep


    def rec_traverse_down(self, word_index, A_p2c, parent_idx, parent_emb, result_table, upward_table):
        result_table[parent_idx] = parent_emb
        if parent_idx in word_index:
            return

        children_idx = np.where(A_p2c[parent_idx] == 1)[0]
        assert len(children_idx) == 2

        parent_context = torch.cat((parent_emb, upward_table[parent_idx]), dim=-1)
        children_emb = torch.tanh(self.decompose(parent_context))
        # children_emb  = self.decompose2(children_emb)
        left_emb, right_emb = torch.split(children_emb, self.hidden_size, dim=-1)
        
        self.rec_traverse_down(word_index, A_p2c, children_idx[0], left_emb, result_table, upward_table)
        self.rec_traverse_down(word_index, A_p2c, children_idx[1], right_emb, result_table, upward_table)

# TODO fix this hot mess -- can we turn the cue into a one-hot embedding even when it's 3D?????
def make_one_hot(x, max_val):
    t = torch.zeros(*x.shape, max_val)
    z = torch.tensor(x).long().unsqueeze(-1)
    t.scatter_(-1, z, value=1)
    return t

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




