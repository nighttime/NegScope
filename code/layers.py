import math
import pdb
import traceback
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from utils import *

class GraphConvTagger(Module):
    def __init__(self, num_layers, input_features, hidden_units, num_classes, vocab_size, directional=False):
        assert num_layers >= 1

        super(GraphConvTagger, self).__init__()

        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, self.hidden_units - self.num_classes)
        # self.convert_to_hidden = nn.Linear(input_features, hidden_units)
        self.graph_conv = GraphConvolution(hidden_units, directional=directional)
        
        # self.layers = nn.ModuleList( \
        #     [GraphConvolution(input_features, hidden_units, directional=directional)] + \
        #     [GraphConvolution(hidden_units, hidden_units, directional=directional) for _ in range(num_layers-1)])
        # self.layer_conversion = GraphConvolution(input_features, hidden_units, directional=directional)
        
        # self.dropout = dropout

        self.classifier = nn.Linear(hidden_units, num_classes)

    def forward(self, x, cue, adj):
        x_emb = self.embedding(torch.tensor(x))
        cue_emb = make_one_hot(cue, self.num_classes)
        # tc = torch.tensor(cue).float().unsqueeze(-1)
        A = torch.FloatTensor(adj)
        x = torch.cat((x_emb, cue_emb), -1)

        # pdb.set_trace()

        # x = F.relu(self.convert_to_hidden(x))
        for _ in range(self.num_layers):
            x = self.graph_conv(x, A)

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
        self.atn_stabilizer = math.sqrt(self.atn_feature_size)
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
                    values = torch.tanh(values)
                    
                    # Convolve neighbors using the score weightings
                    scores = torch.matmul(keys, query)/self.atn_stabilizer
                    atn_weights = F.softmax(scores, dim=-1)

                    # pdb.set_trace()
                    output[b,i] = torch.matmul(atn_weights, values)
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
    def __init__(self, in_size, out_size, num_classes, vocab_size, pos_size, pos_vocab_size, use_pretrained_embs=False, dropout_p=0):
        super(RecurrentTagger, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_classes = num_classes
        self.pos_vocab_size = pos_vocab_size
        self.use_pretrained_embs = use_pretrained_embs

        if self.use_pretrained_embs:
            # self.emb2input = nn.Linear(768, in_size)
            self.lstm_input_size = 768 + num_classes + pos_size
        else:
            self.word_embedding = nn.Embedding(vocab_size, in_size)
            self.lstm_input_size = in_size + num_classes + pos_size

        # self.word_embedding = nn.Embedding(vocab_size, in_size - num_classes)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_size)
        # self.bilstm = nn.LSTM(in_size, out_size, bidirectional=True)
        # self.bilstm = nn.LSTM(in_size+pos_size, out_size, bidirectional=True)
        # self.drop_layer = nn.Dropout(p=0.25)
        self.bilstm = nn.LSTM(self.lstm_input_size, out_size, bidirectional=True)
        self.classifier = nn.Linear(out_size*2, num_classes)

    def forward(self, x, cue, pos, emb):
        bsize = x.shape[0]
        dsize = x.shape[1]

        if self.use_pretrained_embs:
            word_emb = torch.zeros(bsize, dsize, emb[0].shape[1])
            # word_emb = torch.zeros(bsize, dsize, self.in_size)
            for i in range(bsize):
                # e = self.emb2input(torch.tensor(emb[i]).float())
                e = torch.tensor(emb[i]).float()
                word_emb[i,0:e.shape[0],:] += e
        else:
            word_emb = self.word_embedding(torch.tensor(x))
        
        # packed_word_emb = torch.zeros(bsize, dsize, emb[0].shape[1])
        # for i in range(bsize):
        #     e = torch.tensor(emb[i]).float()
        #     packed_word_emb[i,0:e.shape[0],:] += e
        
        # word_emb = self.word_embedding(torch.tensor(x))
        # pos_emb = make_one_hot(pos, self.pos_vocab_size)
        pos_emb = self.pos_embedding(torch.tensor(pos))
        cue_emb = make_one_hot(cue, self.num_classes)

        lstm_input = torch.cat((word_emb, cue_emb, pos_emb), -1)
        # pdb.set_trace()
        # lstm_input = self.drop_layer(lstm_input)

        lstm_output, _ = self.bilstm(lstm_input)

        # class_scores = torch.tanh(lstm_output)
        class_scores = self.classifier(lstm_output)
        class_scores = torch.tanh(class_scores)

        yhat = F.softmax(class_scores, dim=-1)
        # yhat = class_scores

        return yhat


class RecurrentTaggerD(Module):
    def __init__(self, in_size, out_size, num_classes, vocab_size, pos_size, pos_vocab_size, use_pretrained_embs=False, dropout_p=0):
        super(RecurrentTaggerD, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_classes = num_classes
        self.pos_vocab_size = pos_vocab_size
        self.dropout_p = dropout_p
        self.use_pretrained_embs = use_pretrained_embs

        if self.use_pretrained_embs:
            self.emb2input = nn.Linear(768, in_size)
        else:
            self.word_embedding = nn.Embedding(vocab_size, in_size)
        
        self.lstm_input_size = in_size + num_classes + pos_size

        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_size)
        self.lstm_l2r = nn.LSTMCell(self.lstm_input_size, out_size)
        self.lstm_r2l = nn.LSTMCell(self.lstm_input_size, out_size)
        # self.bilstm = nn.LSTM(in_size, out_size, bidirectional=True)
        # self.bilstm = nn.LSTM(in_size+pos_size, out_size, bidirectional=True)
        # self.drop_layer = nn.Dropout(p=0.25)
        # self.bilstm = nn.LSTM(768+num_classes+pos_size, out_size, bidirectional=True)
        self.classifier = nn.Linear(out_size*2, num_classes)

        # TODO: figure out what the unexpected dimension sizing problem is; 
        # run LSTM with dropout (input and hidden-recurrent connections currently being dropped out)

    def forward(self, x, cue, pos, emb):
        bsize = x.shape[0]
        dsize = x.shape[1]
        
        if self.use_pretrained_embs:
            # word_emb = torch.zeros(bsize, dsize, emb[0].shape[1])
            word_emb = torch.zeros(bsize, dsize, self.in_size)
            for i in range(bsize):
                e = self.emb2input(torch.tensor(emb[i]).float())
                # e = torch.tensor(emb[i]).float()
                word_emb[i,0:e.shape[0],:] += e
        else:
            word_emb = self.word_embedding(torch.tensor(x))

        # pos_emb = make_one_hot(pos, self.pos_vocab_size)
        pos_emb = self.pos_embedding(torch.tensor(pos))
        cue_emb = make_one_hot(cue, self.num_classes)

        lstm_input = torch.cat((word_emb, cue_emb, pos_emb), -1)
        in_size = lstm_input.shape[-1]
        # lstm_input = self.drop_layer(lstm_input)

        # lstm_output, _ = self.bilstm(lstm_input)

        l_h = torch.zeros(bsize, self.out_size)
        l_c = torch.zeros(bsize, self.out_size)
        r_h = torch.zeros(bsize, self.out_size)
        r_c = torch.zeros(bsize, self.out_size)
        
        lstm_output = torch.zeros(bsize, dsize, self.out_size*2)

        h_drop_mask = dropout_mask(self.out_size, 0.15)
        # c_drop_mask = dropout_mask(self.hidden_size, 0.3)
        in_drop_mask = dropout_mask(self.lstm_input_size, 0.15)
        
        for i in range(dsize):
            # pdb.set_trace()
            l_h, l_c = self.lstm_l2r(lstm_input[:,0],  (l_h, l_c))
            r_h, r_c = self.lstm_r2l(lstm_input[:,-1], (r_h, r_c))
            if False:
                l_h *= h_drop_mask
                r_h *= h_drop_mask
            # l_c *= c_drop_mask
            # r_c *= c_drop_mask
            lstm_output[:,i,:self.out_size] = l_h
            lstm_output[:,-(i+1),self.out_size:] = r_h
            # pdb.set_trace()

        # class_scores = torch.tanh(lstm_output)
        class_scores = self.classifier(lstm_output)
        class_scores = torch.tanh(class_scores)

        yhat = class_scores

        # yhat = F.softmax(class_scores, dim=-1)

        return yhat


def dropout_mask(size, p):
    drop_mask = torch.ones(size)
    mask_perm = torch.randperm(size)
    drop_idx = int(size*p)
    drop_mask[mask_perm[:drop_idx]] = 0.
    drop_mask[mask_perm] *= 1./(1-p)
    return drop_mask


# WORKING TRNN -- BASIC IMPLEMENTATION WITH NO BELLS OR WHISTLES

# class TreeRecurrentTagger(Module):
#     def __init__(self, hidden_size, num_classes, vocab_size):
#         super(TreeRecurrentTagger, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_classes = num_classes

#         self.word_embedding = nn.Embedding(vocab_size, self.hidden_size-self.num_classes)
#         self.compose = nn.Linear(2*hidden_size, hidden_size)
#         # self.compose2 = nn.Linear(hidden_size, hidden_size)
#         self.decompose = nn.Linear(2*hidden_size, 2*hidden_size)
#         # self.decompose2 = nn.Linear(2*hidden_size, 2*hidden_size)
#         self.global_reverse = nn.Linear(hidden_size, hidden_size)
#         self.classifier = nn.Linear(hidden_size*2, num_classes)

#     def forward(self, x, word_index, cue, adj):
#         assert len(adj.shape) == 4
#         bsize = x.shape[0]
#         dsize = x.shape[1]
#         results = torch.zeros(bsize, dsize, self.num_classes)
        
#         word_emb = self.word_embedding(torch.tensor(x))
#         cue_emb  = make_one_hot(cue, self.num_classes)
#         # cue_emb  = torch.tensor(cue).float().unsqueeze(-1)
#         in_feats = torch.cat((word_emb, cue_emb), -1)

#         # for each sentence in batch
#         for i in range(bsize):
#             # word_emb = self.word_embedding(torch.tensor(x))
#             w_idx = word_index[i]
#             sent = in_feats[i,w_idx]
#             A = adj[i,1]
#             sentlen = A.shape[-1]

#             # traverse up the tree
#             upward_results = torch.zeros(sentlen, self.hidden_size)
#             global_up = self.rec_traverse_up(sent, w_idx, A, 0, upward_results)
#             global_down = torch.tanh(self.global_reverse(global_up))
#             # traverse back down the tree
#             downward_results = torch.zeros(dsize, self.hidden_size)
#             self.rec_traverse_down(w_idx, A, 0, global_down, downward_results, upward_results)

#             # TODO concatenate the upward and downward results before classification
#             tree_data = torch.cat((downward_results, upward_results), dim=-1)
#             classifications = self.classifier(tree_data)
#             class_scores = torch.sigmoid(classifications)

#             yhat = F.softmax(class_scores, dim=-1)

#             results[i] = yhat
#             # pdb.set_trace()

#         return results

#     def rec_traverse_up(self, x_emb, word_index, A_p2c, parent_idx, result_table):
#         def get_child(idx):
#             if idx in word_index:
#                 locs = np.where(word_index==idx)[0]
#                 assert len(locs) == 1
#                 return x_emb[locs[0]]
#             else:
#                 return self.rec_traverse_up(x_emb, word_index, A_p2c, idx, result_table)

#         children_idx = np.where(A_p2c[parent_idx] == 1)[0]
#         assert len(children_idx) in (0, 2)

#         left  = get_child(children_idx[0])
#         right = get_child(children_idx[1])

#         parent_context = torch.cat((left, right), dim=-1)
#         parent_rep = torch.tanh(self.compose(parent_context))
#         # parent_rep  = torch.sigmoid(self.compose2(parent_rep1))

#         result_table[parent_idx] = parent_rep

#         return parent_rep


#     def rec_traverse_down(self, word_index, A_p2c, parent_idx, parent_emb, result_table, upward_table):
#         result_table[parent_idx] = parent_emb
#         if parent_idx in word_index:
#             return

#         children_idx = np.where(A_p2c[parent_idx] == 1)[0]
#         assert len(children_idx) == 2

#         parent_context = torch.cat((parent_emb, upward_table[parent_idx]), dim=-1)
#         children_emb = torch.tanh(self.decompose(parent_context))
#         # children_emb  = self.decompose2(children_emb)
#         left_emb, right_emb = torch.split(children_emb, self.hidden_size, dim=-1)
        
#         self.rec_traverse_down(word_index, A_p2c, children_idx[0], left_emb, result_table, upward_table)
#         self.rec_traverse_down(word_index, A_p2c, children_idx[1], right_emb, result_table, upward_table)



class TreeRecurrentTagger(Module):
    def __init__(self, hidden_size, num_classes, vocab_size, syntax_size, use_pretrained_embs=False, dropout_p=0):
        super(TreeRecurrentTagger, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_syntax_labels = syntax_size
        self.hidden_syn_size = 50
        self.use_pretrained_embs = use_pretrained_embs

        if not self.use_pretrained_embs:
            self.word_embedding = nn.Embedding(vocab_size, self.hidden_size)
            self.leaf2node = nn.Linear(self.hidden_size + self.hidden_syn_size + self.num_classes, self.hidden_size)
        else:
            self.leaf2node = nn.Linear(768 + self.hidden_syn_size + self.num_classes, self.hidden_size)
                
        self.syn_embedding = nn.Embedding(self.num_syntax_labels, self.hidden_syn_size)
        
        self.compose = nn.Linear(2*self.hidden_size + self.hidden_syn_size, self.hidden_size)
        self.global_reverse = nn.Linear(self.hidden_size + self.hidden_syn_size, self.hidden_size)
        self.decompose = nn.Linear(2*self.hidden_size + 2*self.hidden_syn_size, 2*self.hidden_size)
        
        self.classifier = nn.Linear(2*self.hidden_size, self.num_classes)

    def forward(self, x, word_index, cue, adj, pos, pre_embs):
        assert len(adj.shape) == 4
        bsize = x.shape[0]
        dsize = x.shape[1]
        results = torch.zeros(bsize, dsize, self.num_classes)
        x = torch.tensor(x)

        self.word_drop_mask = dropout_mask(768 if self.use_pretrained_embs else self.hidden_size, 0.5)
        self.syn_drop_mask  = dropout_mask(self.hidden_syn_size, 0.5)
        self.leaf_drop_mask = dropout_mask(self.hidden_size,     0.5)
        self.up_drop_mask   = dropout_mask(self.hidden_size,     0.2)
        self.rev_drop_mask  = dropout_mask(self.hidden_size,     0.5)
        self.down_drop_mask = dropout_mask(2*self.hidden_size,   0.2)
        self.tree_drop_mask = dropout_mask(2*self.hidden_size,   0.5)

        # LR=0.001 / 0.7688 test f1 / 0.9067 acc /@epoch 86 (ce loss)
        # self.syn_drop_mask  = dropout_mask(self.hidden_syn_size, 0.5)
        # self.word_drop_mask = dropout_mask(self.hidden_size,     0.5)
        # self.up_drop_mask   = dropout_mask(self.hidden_size,     0.2)
        # self.rev_drop_mask  = dropout_mask(self.hidden_size,     0.5)
        # self.down_drop_mask = dropout_mask(2*self.hidden_size,   0.2)
        # self.tree_drop_mask = dropout_mask(2*self.hidden_size,   0.5)

        # LR=0.001 / 0.7154 test f1 / 0.8427 acc @epoch34 (f1 loss)
        # self.syn_drop_mask  = dropout_mask(self.hidden_syn_size, 0.4)
        # self.word_drop_mask = dropout_mask(self.hidden_size,     0.4)
        # self.up_drop_mask   = dropout_mask(self.hidden_size,     0.2)
        # self.rev_drop_mask  = dropout_mask(self.hidden_size,     0.2)
        # self.down_drop_mask = dropout_mask(2*self.hidden_size,   0.2)
        # self.tree_drop_mask = dropout_mask(2*self.hidden_size,   0.2)

        # for each sentence in batch
        for i in range(bsize):
            w_idx = word_index[i]
            
            tree_tokens = x[i]
            s = tree_tokens[w_idx]
            c = cue[i][w_idx]
            p = torch.tensor(pos[i][:len(s)])
            
            if self.use_pretrained_embs:
                word_emb = torch.tensor(pre_embs[i]).float()
            else:
                word_emb = self.word_embedding(s)

            cue_emb = make_one_hot(c, self.num_classes)
            # pos_emb  = make_one_hot(p, self.num_syntax_labels)
            pos_emb = self.syn_embedding(p)
            if self.training:
                word_emb = word_emb.clone() * self.word_drop_mask
                pos_emb = pos_emb.clone() * self.syn_drop_mask

            in_emb  = torch.cat((word_emb, cue_emb, pos_emb), -1)
            x_emb = self.leaf2node(in_emb)

            if self.training:
                x_emb = x_emb.clone() * self.leaf_drop_mask

            A = adj[i,1]
            sentlen = A.shape[-1]


            # traverse up the tree
            upward_results = torch.zeros(sentlen, self.hidden_size)
            upward_results[w_idx] = x_emb
            global_up_emb = self.rec_traverse_up(tree_tokens, w_idx, A, 0, upward_results)
            if self.training:
                global_up_emb = global_up_emb.clone() * self.rev_drop_mask

            # root_constituent = make_one_hot(tree_tokens[0], self.num_syntax_labels)
            root_constituent = self.syn_embedding(tree_tokens[0])
            if self.training:
                root_constituent = root_constituent.clone() * self.syn_drop_mask
            root_up_emb = torch.cat((global_up_emb, root_constituent), dim=-1)
            global_down_emb = torch.tanh(self.global_reverse(root_up_emb))

            # traverse back down the tree
            downward_results = torch.zeros(dsize, self.hidden_size)
            self.rec_traverse_down(tree_tokens, pos_emb, w_idx, A, 0, global_down_emb, downward_results, upward_results)

            # concatenate the upward and downward results before classification
            tree_data = torch.cat((downward_results, upward_results), dim=-1)
            if self.training:
                tree_data = tree_data.clone() * self.tree_drop_mask
            classifications = self.classifier(tree_data)
            class_scores = torch.tanh(classifications)

            # yhat = F.softmax(class_scores, dim=-1)
            yhat = class_scores

            results[i] = yhat

        return results

    def rec_traverse_up(self, tree_tokens, word_index, A_p2c, parent_idx, result_table):
        def get_child(idx):
            if idx in word_index:
                return result_table[idx]
            else:
                return self.rec_traverse_up(tree_tokens, word_index, A_p2c, idx, result_table)

        children_idx = np.where(A_p2c[parent_idx] == 1)[0]
        assert len(children_idx) == 2

        left  = get_child(children_idx[0])
        right = get_child(children_idx[1])

        if self.training:
            left = left.clone() * self.up_drop_mask
            right = right.clone() * self.up_drop_mask

        # constituent = make_one_hot(tree_tokens[parent_idx], self.num_syntax_labels)
        constituent = self.syn_embedding(tree_tokens[parent_idx])
        upward_context = torch.cat((constituent, left, right), dim=-1)
        parent_emb = torch.tanh(self.compose(upward_context))

        result_table[parent_idx] = parent_emb

        return parent_emb

    def rec_traverse_down(self, tree_tokens, pos_emb, word_index, A_p2c, parent_idx, parent_emb, result_table, upward_table):
        result_table[parent_idx] = parent_emb
        if parent_idx in word_index:
            return

        children_idx = np.where(A_p2c[parent_idx] == 1)[0]
        assert len(children_idx) == 2
        l_idx, r_idx = children_idx
        
        def get_child_syntax(idx):
            match = (word_index == idx).nonzero()[0].squeeze()
            if match.size  == 1:
                return pos_emb[match]
            elif match.size == 0:
                return self.syn_embedding(tree_tokens[idx])

        cons_left  = get_child_syntax(l_idx)
        cons_right = get_child_syntax(r_idx)

        if self.training:
            cons_left  = cons_left.clone()  * self.syn_drop_mask
            cons_right = cons_right.clone() * self.syn_drop_mask

        # constituent = make_one_hot(tree_tokens[parent_idx], self.num_syntax_labels)
        # constituent = self.syn_embedding(tree_tokens[parent_idx])

        parent_context = torch.cat((parent_emb, upward_table[parent_idx]), dim=-1)
        if self.training:
            parent_context = parent_context.clone() * self.down_drop_mask
        downward_context = torch.cat((cons_left, cons_right, parent_context), dim=-1)
        # parent_context = self.drop_layer(parent_context)
        children_emb = torch.tanh(self.decompose(downward_context))

        left_emb, right_emb = torch.split(children_emb, self.hidden_size, dim=-1)

        self.rec_traverse_down(tree_tokens, pos_emb, word_index, A_p2c, children_idx[0], left_emb, result_table, upward_table)
        self.rec_traverse_down(tree_tokens, pos_emb, word_index, A_p2c, children_idx[1], right_emb, result_table, upward_table)




class TreeLSTMTagger(Module):
    def __init__(self, hidden_size, num_classes, vocab_size, syntax_size, use_pretrained_embs=False, dropout_p=0):
        super(TreeLSTMTagger, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_syntax_labels = syntax_size
        self.hidden_syn_size = 50
        self.use_pretrained_embs = use_pretrained_embs

        if not self.use_pretrained_embs:
            self.word_embedding = nn.Embedding(vocab_size, self.hidden_size)
            self.leaf2node = nn.Linear(self.hidden_size + self.hidden_syn_size + self.num_classes, self.hidden_size)
        else:
            self.leaf2node = nn.Linear(768 + self.hidden_syn_size + self.num_classes, self.hidden_size)
                
        self.syn_embedding = nn.Embedding(self.num_syntax_labels, self.hidden_syn_size)
        self.drop_layer = nn.Dropout(p=dropout_p)
        
        self.compose = nn.LSTMCell(self.hidden_syn_size, self.hidden_size)
        self.global_reverse_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.global_reverse_c = nn.Linear(self.hidden_size, 2*self.hidden_size)
        self.decompose = nn.LSTMCell(2*self.hidden_syn_size, 2*self.hidden_size)
        
        self.classifier = nn.Linear(2*self.hidden_size, self.num_classes)

    def forward(self, x, word_index, cue, adj, pos, pre_embs):
        assert len(adj.shape) == 4
        bsize = x.shape[0]
        dsize = x.shape[1]
        results = torch.zeros(bsize, dsize, self.num_classes)
        x = torch.tensor(x)

        # for each sentence in batch
        for i in range(bsize):
            w_idx = word_index[i]
            
            tree_tokens = x[i]
            s = tree_tokens[w_idx]
            c = cue[i][w_idx]
            p = torch.tensor(pos[i][:len(s)])
            
            if self.use_pretrained_embs:
                word_emb = torch.tensor(pre_embs[i]).float()
            else:
                word_emb = self.word_embedding(s)
            cue_emb  = make_one_hot(c, self.num_classes)
            # pos_emb  = make_one_hot(p, self.num_syntax_labels)
            pos_emb  = self.syn_embedding(p)
            in_emb   = torch.cat((word_emb, cue_emb, pos_emb), -1)
            x_emb = self.leaf2node(in_emb)

            A = adj[i,1]
            sentlen = A.shape[-1]

            # traverse up the tree
            upward_results = torch.zeros(sentlen, self.hidden_size)
            upward_results[w_idx] = x_emb
            up_h, up_c = self.rec_traverse_up(tree_tokens, w_idx, A, 0, upward_results)
            # root_constituent = make_one_hot(tree_tokens[0], self.num_syntax_labels)
            # root_constituent = self.syn_embedding(tree_tokens[0])
            # root_up_emb = torch.cat((global_up_emb, root_constituent), dim=-1)
            # root_up_emb = self.drop_layer(root_up_emb)
            down_half_h = torch.tanh(self.global_reverse_h(up_h))
            down_c = torch.tanh(self.global_reverse_c(up_c))
            # traverse back down the tree
            downward_results = torch.zeros(dsize, self.hidden_size)
            self.rec_traverse_down(tree_tokens, pos_emb, w_idx, A, 0, down_half_h, down_c, downward_results, upward_results)

            # concatenate the upward and downward results before classification
            tree_data = torch.cat((downward_results, upward_results), dim=-1)
            # tree_data = self.drop_layer(tree_data)
            classifications = self.classifier(tree_data)
            class_scores = torch.tanh(classifications)

            yhat = F.softmax(class_scores, dim=-1)

            results[i] = yhat
            # pdb.set_trace()

        return results

    def rec_traverse_up(self, tree_tokens, word_index, A_p2c, parent_idx, result_table):
        def get_child(idx):
            if idx in word_index:
                return result_table[idx].unsqueeze(0), torch.zeros(self.hidden_size).unsqueeze(0)
            else:
                return self.rec_traverse_up(tree_tokens, word_index, A_p2c, idx, result_table)

        children_idx = np.where(A_p2c[parent_idx] == 1)[0]
        assert len(children_idx) == 2

        l_h, l_c = get_child(children_idx[0])
        r_h, r_c = get_child(children_idx[1])

        # constituent = make_one_hot(tree_tokens[parent_idx], self.num_syntax_labels)
        parent_target = self.syn_embedding(tree_tokens[parent_idx]).unsqueeze(0)
        # parent_context = torch.cat((constituent, left, right), dim=-1)
        # parent_context = self.drop_layer(parent_context)
        children_h = l_h + r_h
        children_c = l_c + r_c
        parent_h, parent_c = self.compose(parent_target, (children_h, children_c))
        
        result_table[parent_idx] = parent_h

        return parent_h, parent_c

    def rec_traverse_down(self, tree_tokens, pos_emb, word_index, A_p2c, parent_idx, parent_half_h, parent_c, result_table, upward_table):
        result_table[parent_idx] = parent_half_h
        if parent_idx in word_index:
            return

        children_idx = np.where(A_p2c[parent_idx] == 1)[0]
        assert len(children_idx) == 2
        l_idx, r_idx = children_idx
        
        def get_child_syntax(idx):
            match = (word_index == idx).nonzero()[0].squeeze()
            if match.size  == 1:
                return pos_emb[match]
            elif match.size == 0:
                return self.syn_embedding(tree_tokens[idx])

        cons_left  = get_child_syntax(l_idx)
        cons_right = get_child_syntax(r_idx)

        child_targets = torch.cat((cons_left, cons_right), dim=-1).unsqueeze(0)
        parent_h = torch.cat((parent_half_h, upward_table[parent_idx].unsqueeze(0)), dim=-1)
        # parent_context = self.drop_layer(parent_context)
        children_h, children_c = self.decompose(child_targets, (parent_h, parent_c))

        left_half_h, right_half_h = torch.split(children_h, self.hidden_size, dim=-1)

        self.rec_traverse_down(tree_tokens, pos_emb, word_index, A_p2c, l_idx, left_half_h, children_c, result_table, upward_table)
        self.rec_traverse_down(tree_tokens, pos_emb, word_index, A_p2c, r_idx, right_half_h, children_c, result_table, upward_table)



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
        return 1 - F1.mean()




