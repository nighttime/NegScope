from data import *
from nsd import *
from graphviz import Digraph
import torch
from torch.autograd import Variable, Function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import sys
from collections import Counter

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

def print_f1():
    tp1, fp1, fn1 = 833, 78, 127 #821, 49, 139
    tp2, fp2, fn2 = 712, 74, 133 #684, 72, 161
    tp = float(tp1+tp2)
    fp = float(fp1+fp2)
    fn = float(fn1+fn2)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = (2*p*r)/(p+r)
    print('p: {:.4f}  r: {:.4f}  f1: {:.4f}'.format(p, r, f1))


def np_acc(actual, expected):
    return (actual == expected).mean()

def np_f1_score(actual, expected):
    # pdb.set_trace()
    tp = np.sum(expected * actual, axis=-1)
    tn = np.sum((1.0-expected) * (1.0-actual), axis=-1)
    fp = np.sum((1.0-expected) * actual, axis=-1)
    fn = np.sum(expected * (1 - actual), axis=-1)

    p, r, f1 = np_calc_f1(tp, fp, fn)
    return p, r, f1

def np_calc_f1(tp, fp, fn):
    precision = tp / (tp + fp + EPSILON)
    recall    = tp / (tp + fn + EPSILON)
    f1 = (2 * precision * recall) / (precision + recall + EPSILON)
    return precision, recall, f1

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

    train, dev, test = format_data(
        corpora, 
        word2ind, 
        syn2ind, 
        directional=directional, 
        row_normalize=row_normalize, 
        embs_folder='pretrained_embs')

    dev_scope = dev[5]

    trees = corpora[2]
    A, ts, pos, word_index, cue, scope, embs = test

    colorings = np.load(sys.argv[1])

    def qt(n):
        #true_scope, cue, colorings=None, pos_branches=False, fig=False):
        return trees[n].print_qtree(scope[n], cue[n][word_index[n]], colorings=colorings[n], fig=True)

    def pqt(n):
        print(qt(n))

    results = [(np_acc(colorings[i][word_index[i]], scope[i]), *np_f1_score(colorings[i][word_index[i]], scope[i]), len(t.constituents), t, i) for i,t in enumerate(trees)]
    # results = [(f, l, i) for a, p, r, f, l, t, i in results]
    sorted_res = sorted(results, key=lambda x: x[0], reverse=True)

    res_idx = [(a, f, l, i) for a, p, r, f, l, t, i in sorted_res]

    sorted_accs = [j[0]*100 for j in sorted_res]

    poor_res = [j for j in sorted_res if j[0] <= 0.67]
    poor_exs = [qt(j[-1]) + '\n%num={}'.format(j[-1]) + '\n\n' for j in poor_res]
    # for ex in poor_exs:
    #     print(ex)

    good_exs = [qt(j[-1]) + '\n%num={}'.format(j[-1]) + '\n\n' for j in sorted_res if j[0] > 0.99]
    best_exs = [j for j in sorted_res if j[0] == 1]
    # for ex in good_exs:
        # print(ex)

    # scope_means = [np.mean(dev_scope[n]) for n in range(len(dev_scope))]
    # smm = np.mean(scope_means)

    accs = [a for a, p, r, f, l, t, i in results]
    cons = [l for a, p, r, f, l, t, i in results]
    tree_depths = [t.max_tree_depth() for a, p, r, f, l, t, i in results]
    sent_lens = [len(t.words) for a, p, r, f, l, t, i in results]
    complexity_ratios = [tree_depths[i]/sent_lens[i] for i in range(len(tree_depths))]

    def get_phrase(const, t):
        if const.is_leaf():
            return t.words[const.leaf_ref]
        else:
            return ' '.join(get_phrase(c, t) for c in const.children)

    ct_neg_children = 0
    ct_neg_parent = 0
    ct_pos_parent = 0
    blocked_ctr = Counter()
    vp_mod = []
    np_mod = []
    clause = []
    for a, p, r, f, l, t, i in results:
        if a > 0:
            for const in t.constituents:
                if not const.is_leaf():
                    if not colorings[i][const.traversal_idx]:
                        continue
                    if sum(colorings[i][c.traversal_idx] for c in const.children) != 1:
                        continue
                    child = [c for c in const.children if not colorings[i][c.traversal_idx]][0]
                    key = t.words[child.leaf_ref] if child.is_leaf() else child.constituent
                    blocked_ctr[key] += 1
                    if key == '<S\\NP>\\<S\\NP>':
                        vp_mod.append(get_phrase(child, t))
                    elif key == 'NP\\NP':
                        np_mod.append(get_phrase(child, t))
                    elif key == 'S[dcl]\\NP':
                        clause.append(get_phrase(child, t))

    

    # bin_width = 5
    # ax = sns.distplot(all_accs, kde=False, bins=range(0,101,bin_width))
    # plt.xticks(range(0,101,10))
    # plt.axvline(x=67, color='k', linestyle='--', linewidth=1)
    # ax.set_ylabel('Number of Sentences')
    # ax.set_xlabel('Prediction Accuracy')
    # ax.set_title('Histogram of Token Prediction Accuracy per Sentence')
    # plt.legend(['Random Prediction Baseline'])

    # plt.show()

    

    pdb.set_trace()


def main():
    # print_f1()
    examine_results()







if __name__ == '__main__':
    main()



