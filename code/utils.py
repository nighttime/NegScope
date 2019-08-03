from data import *
from graphviz import Digraph
import torch
from torch.autograd import Variable, Function
import numpy as np
import pdb
import sys

class Color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

EPSILON = 1e-10


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def viz_register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        grad_output = grad_output.data
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot

# get_dot = register_hooks(z)
# z.backward()
# dot = get_dot()
# dot.save('tmp.dot')

def compute_f1():
    tp1, fp1, fn1 = 960, 1741, 0 #821, 49, 139
    tp2, fp2, fn2 = 845, 1302, 0 #684, 72, 161
    tp = float(tp1+tp2)
    fp = float(fp1+fp2)
    fn = float(fn1+fn2)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = (2*p*r)/(p+r)
    print('p: {:.4f}  r: {:.4f}  f1: {:.4f}'.format(p, r, f1))

def examine_results():
    syntax_folder = '../starsem-st-2012-data/cd-sco-CCG/corpus/'
    directional = True
    row_normalize = False
    pos_from_syntax=True
    condense_trees = True
    pretrained_embs = True

    # Retrieve data
    corpora, corp_sent_copies, word2ind, syn2ind, full_vocab = get_parse_data(
        only_negations=True, 
        external_syntax_folder=syntax_folder,
        derive_pos_from_syntax=pos_from_syntax,
        condense_single_branches=condense_trees)

    _, _, test = format_data(
        corpora, 
        word2ind, 
        syn2ind, 
        directional=directional, 
        row_normalize=row_normalize, 
        embs_folder='pretrained_embs')

    A, ts, pos, word_index, cue, scope, embs = test

    colorings = np.load(sys.argv[1])
    pdb.set_trace()


def main():
    # compute_f1()
    examine_results()







if __name__ == '__main__':
    main()



