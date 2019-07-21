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


def _starsem_sentence_vals(fname):
	# sents : [[sent, pos, syntax, [neg_scope1, neg_scope2]]]
	sents = []
	new_sent = True

	with open(fname) as f:
		for line in f:
			if line == '\n':
				new_sent = True
				continue

			if new_sent:
				# sentence format : [words, pos, syntax, [neg_scope1, neg_scope2]]
				sents.append([list(), list(), str(), [list(), list()]])
				new_sent = False

			# l : [chapter, sent_num, tok_num, word, lemma, pos, syntax, [neg_trig, scope, event]{0,2} ]
			l = line.split('\t')

			word, pos, syntax = l[3], l[5], l[6]

			# Read parse tree data
			sents[-1][0].append(word)
			sents[-1][1].append(pos)
			sents[-1][2] += syntax
			
			# Read negation data
			if not '***' in l[-1]:
				n1 = [None if x == '_' else x for x in l[7:10]]
				sents[-1][3][0].append(n1)
				if len(l) > 10:
					n2 = [None if x == '_' else x for x in l[10:13]]
					sents[-1][3][1].append(n2)

		return sents

def read_starsem_scope_data(fname, external_syntax_fname=None):
	''' Read in training data, parse sentence syntax, and construct consituent trees'''

	# Read in annotation data from corpus
	sents = _starsem_sentence_vals(fname)

	# Remove sentences that are longer than 70 words (cannot be parsed using Easy CCG)
	sents = [x for x in sents if len(x[0]) <= 70]

	# Construct constituent trees from input data, duplicating sentences if they have 2 negations

	# sent_trees: [ParsedSentence]
	sent_trees = []

	# if using external syntax (ie CCG)
	if external_syntax_fname:
		external_syntax_trees = list(open(external_syntax_fname))

	ext_p = 0
	for i,vals in enumerate(sents):
		syntax = external_syntax_trees[ext_p].replace('\n', '') if external_syntax_fname else vals[2]
		
		if syntax.count('*') != len(vals[0]):
			print('No ccg parse. Skipping:',' '.join(vals[0]))
			continue

		if vals[3][1]:
		# 2 negations in this sentence
			sent_trees.append(ParsedSentence(*vals[:2], syntax, negation=vals[3][0]))
			sent_trees.append(ParsedSentence(*vals[:2], syntax, negation=vals[3][1]))
		elif vals[3][0]:
		# 1 negation in this sentence
			sent_trees.append(ParsedSentence(*vals[:2], syntax, negation=vals[3][0]))
		else:
		# 0 negations in this sentence
			sent_trees.append(ParsedSentence(*vals[:2], syntax))

		ext_p += 1

	return sent_trees


def _format_dataset(dataset, maxlen):
	# input: a corpus : [(A, ts, mask, cue, scope)]
	# output: reformatted as numpy arrays : (A, ts, mask, cue, scope)
	dlen = len(dataset)
	Ashape = dataset[0][0].shape

	# A            = np.zeros((dlen, maxlen, maxlen))
	# adjust shape of A in case it's a directional matrix (will have extra indice)
	A            = np.zeros(tuple([dlen] + list(Ashape[0:(len(Ashape)-2)]) + [maxlen, maxlen]))
	ts           = np.zeros((dlen, maxlen), dtype=int)
	pos          = np.zeros((dlen, maxlen), dtype=int)
	word_index   = np.zeros(dlen, dtype=object)
	cue          = np.zeros((dlen, maxlen))
	scope        = np.zeros(dlen, dtype=object)

	for i,d in enumerate(dataset):
		ds = d[0].shape
		# pdb.set_trace()
		if len(ds) == 3:
			A[i,:,0:ds[1],0:ds[2]] += d[0]
		else:
			A[i,0:ds[0],0:ds[1]] += d[0]
		ts[i,0:len(d[1])]      += np.array(d[1])
		pos[i,0:len(d[2])]     += np.array(d[2])
		word_index[i]           = np.array([i for i,v in enumerate(d[3]) if v])
		cue[i,0:len(d[3])]     += np.array(d[4])
		scope[i]                = np.array(d[5])

	return (A, ts, pos, word_index, cue, scope)


def format_data(corpora, word2ind, syn2ind, directional=False, row_normalize=True):
	# input: a corpora : [[ParseTree]]
	# output: reformatted data of type : [(A, ts, mask, cue, scope)]
	data_splits = []
	for corpus in corpora:
		d = []
		for s in corpus:
			A, toks, word_mask = s.adjacency_matrix(directional=directional, row_normalize=row_normalize)
			ts = [word2ind.get(t, UNK_TOK) if word_mask[i] else syn2ind.get(t, UNK_TOK) for i,t in enumerate(toks)]
			pos = [syn2ind.get(p, UNK_TOK) for p in s.pos]
			cue = s.negation_cue()
			scope = s.negation_surface_scope()
			d.append((A, ts, pos, word_mask, cue, scope))
		data_splits.append(d)

	maxlen = max(max(len(s[1]) for s in d) for d in data_splits)
	return [_format_dataset(d, maxlen) for d in data_splits]

def read_corpora(only_words=False, only_negations=False, external_syntax_folder=None, derive_pos_from_syntax=False, condense_single_branches=False):
	# output: list of corpora, each a list of ParseTree : [[ParseTree]] OR a list of words : [[str]]
	corpora = []
	for corpus_file in [TRAINING_FILE, DEV_FILE, [TEST_FILE_A, TEST_FILE_B]]:
		corpus = []
		if isinstance(corpus_file, list):
			for c in corpus_file:
				print('reading in:', c)
				ext_syntax = external_syntax_folder + c if external_syntax_folder else None
				if only_words:
					corpus += _starsem_sentence_vals(DATA_FOLDER + c)
				else:
					corpus += read_starsem_scope_data(DATA_FOLDER + c, external_syntax_fname=ext_syntax)
		else:
			print('reading in:', corpus_file)
			ext_syntax = external_syntax_folder + corpus_file if external_syntax_folder else None
			if only_words:
				corpus = _starsem_sentence_vals(DATA_FOLDER + corpus_file)
			else:
				corpus = read_starsem_scope_data(DATA_FOLDER + corpus_file, external_syntax_fname=ext_syntax)
		
		if only_negations:
			if only_words:
				# Keep sentence values if the negation component contains a scope
				corpus = [vs for vs in corpus if len(vs[-1][0])]
			else:
				corpus = [s for s in corpus if s.negation]

		if only_words:
			corpus = [vs[0] for vs in corpus]
		else:
			if derive_pos_from_syntax:
				for t in corpus:
					t.pos = t.extract_pos_from_syntax()
			if condense_single_branches:
				for t in corpus:
					t.condense_single_branches()

		corpora.append(corpus)
		
	return corpora

def make_corpus_index(corpus):
	# input: a corpus, as a list of ParseTree : [ParseTree]
	# output: a mapping of unique words to unique integers, with additional UNK and PAD tokens
	words = set(w for s in corpus for w in s.tree_leaf_tokens())
	cons  = set(c for s in corpus for c in s.tree_node_tokens())
	poses = set(tag for s in corpus for tag in s.pos)
	cons |= poses

	return _make_index(words), _make_index(cons)

def _make_index(vocab):
	word2ind = {w:i+2 for i,w in enumerate(vocab)}
	word2ind.update({'UNK':UNK_TOK, 'PAD':PAD_TOK})
	return word2ind

def get_word_data():
	return read_corpora(only_words=True)


def get_parse_data(only_negations=False, external_syntax_folder=None, derive_pos_from_syntax=False, condense_single_branches=False):
	# Read in corpus data
	corpora = read_corpora(
		only_negations=only_negations, 
		external_syntax_folder=external_syntax_folder, 
		derive_pos_from_syntax=derive_pos_from_syntax,
		condense_single_branches=condense_single_branches)

	# Build vocabulary from training data
	word2ind, syn2ind = make_corpus_index(corpora[0])

	return corpora, word2ind, syn2ind





def main():
	A_c2p = np.zeros((7,7))
	c2p = [(0,4),(1,4),(2,5),(3,5),(4,6),(5,6)]
	for pair in c2p:
		A_c2p[pair] = 1
	leaves = [1,1,1,1,0,0,0]
	# pdb.set_trace()
	a = adj_mat_to_binary_combs(A_c2p, A_c2p.T, leaves)
	pdb.set_trace()

if __name__ == '__main__':
	main()








