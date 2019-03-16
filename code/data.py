from collections import defaultdict
import numpy as np
import torch
from torch.nn.modules import *
import pdb

from ParsedSentence import *

DATA_FOLDER = '../starsem-st-2012-data/cd-sco/corpus/'
TRAINING_FILE = 'training/SEM-2012-SharedTask-CD-SCO-training-09032012.txt'
DEV_FILE = 'dev/SEM-2012-SharedTask-CD-SCO-dev-09032012.txt'
TEST_FILE_A = 'test-gold/SEM-2012-SharedTask-CD-SCO-test-cardboard-GOLD.txt'
TEST_FILE_B = 'test-gold/SEM-2012-SharedTask-CD-SCO-test-circle-GOLD.txt'

UNK_TOK = 1
PAD_TOK = 0

def read_starsem_scope_data(fname):
	''' Read in training data, parse sentence syntax, and construct consituent trees'''

	# Read in annotation data from corpus
	
	# {chapter_name: {sentence_num: [sent, pos, syntax, [neg_scope1, neg_scope2]}}
	chapters = defaultdict(lambda: defaultdict(lambda: [list(), list(), str(), [list(), list()]]))

	with open(fname) as f:
		for line in f:
			if len(line) < 2:
				continue

			l = line.split('\t')
			# [chapter, sen_num, tok_num, word, lemma, pos, syntax, [neg_trig, scope, event]{0,2} ]

			ch, sen_num = l[0], int(l[1])

			# Read parse tree data
			chapters[ch][sen_num][0].append(l[3])
			chapters[ch][sen_num][1].append(l[5])
			chapters[ch][sen_num][2] += l[6]
			
			# Read negation data
			if not '***' in l[-1]:
				n1 = [None if x == '_' else x for x in l[7:10]]
				chapters[ch][sen_num][3][0].append(n1)
				if len(l) > 10:
					n2 = [None if x == '_' else x for x in l[10:13]]
					chapters[ch][sen_num][3][1].append(n2)
				

	# Simplify chapter data

	# {chapter_name: [sent_vals]}
	chapter_sents = {name: [vals for _,vals in sorted(sents.items(), key=lambda x: x[0])] for name,sents in chapters.items()}

	# Construct constituent trees from input data, duplicating sentences if they have 2 negations

	# {chapter_name: [ParsedSentence]}
	chapter_trees = {}
	for name,corpus in chapter_sents.items():
		sents = []
		for vals in corpus:
			# 2 negations in this sentence
			if vals[3][1]:
				sents.append(ParsedSentence(*vals[:3], vals[3][0]))
				sents.append(ParsedSentence(*vals[:3], vals[3][1]))
			# 1 negation in this sentence
			elif vals[3][0]:
				sents.append(ParsedSentence(*vals[:3], vals[3][0]))
			# 0 negations in this sentence
			else:
				sents.append(ParsedSentence(*vals[:3]))
		chapter_trees[name] = sents

	return chapter_trees


def _format_dataset(dataset, maxlen):
	# dataset = [(A, ts, mask, cue, scope)]
	dlen = len(dataset)

	A            = np.zeros((dlen, maxlen, maxlen))
	ts           = np.zeros((dlen, maxlen), dtype=int)
	word_index   = np.zeros(dlen, dtype=object)
	cue          = np.zeros((dlen, maxlen))
	scope        = np.zeros(dlen, dtype=object)

	newset = (A, ts, word_index, cue, scope)

	for i,d in enumerate(dataset):
		# pdb.set_trace()
		A[i,0:d[0].shape[0],0:d[0].shape[1]] += d[0]
		ts[i,0:len(d[1])]                    += np.array(d[1])
		word_index[i]                            = np.array([i for i,v in enumerate(d[2]) if v])
		cue[i,0:len(d[3])]                   += np.array(d[3])
		scope[i]                              = np.array(d[4])

	return newset

def format_data(train_unpacked, dev_unpacked, test_unpacked):
	# data format : [(A, ts, mask, cue, scope)]
	maxlen = max(max(len(x[1]) for x in d) for d in [train_unpacked, dev_unpacked, test_unpacked])
	
	train = _format_dataset(train_unpacked, maxlen)
	dev = _format_dataset(dev_unpacked, maxlen)
	test = _format_dataset(test_unpacked, maxlen)

	return train, dev, test
			
def get_data(only_negations=False):
	# Read in corpus data
	corpora = []
	for corpus_file in [TRAINING_FILE, DEV_FILE, [TEST_FILE_A, TEST_FILE_B]]:
		chapters = {}
		if isinstance(corpus_file, list):
			for c in corpus_file:
				print('reading in:', c)
				chapters.update(read_starsem_scope_data(DATA_FOLDER + c))
		else:
			print('reading in:', corpus_file)
			chapters.update(read_starsem_scope_data(DATA_FOLDER + corpus_file))
		
		sents = [sent for _,chap in chapters.items() for sent in chap]
		if only_negations:
			sents = [s for s in sents if s.negation]
		corpora.append(sents)

	# Build vocabulary from training data
	all_cons = [(s.tree_node_tokens(), s.tree_leaf_tokens()) for s in corpora[0]]
	sen_cons, sen_words = tuple(zip(*all_cons))
	cons_list, words_list = [x for s in sen_cons for x in s], [w for s in sen_words for w in s]
	cons, words = set(cons_list), set(words_list)
	vocab = cons | words

	word2ind = {w:i+2 for i,w in enumerate(vocab)}
	word2ind.update({'UNK':UNK_TOK, 'PAD':PAD_TOK})

	# Format corpus data for the GCN
	data_splits = []
	for corpus in corpora:
		d = [(s.adjacency_matrix(), s.negation_cue(), s.negation_surface_scope()) for s in corpus]
		d = [(A, [word2ind.get(w, UNK_TOK) for w in ts], word_mask, cue, scope) for ((A, ts, word_mask), cue, scope) in d]
		data_splits.append(d)

	return data_splits, word2ind






