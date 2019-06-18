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

# def read_starsem_scope_data(fname):
# 	''' Read in training data, parse sentence syntax, and construct consituent trees'''

# 	# Read in annotation data from corpus
	
# 	# {chapter_name: {sentence_num: [sent, pos, syntax, [neg_scope1, neg_scope2]}}
# 	chapters = defaultdict(lambda: defaultdict(lambda: [list(), list(), str(), [list(), list()]]))

# 	with open(fname) as f:
# 		for line in f:
# 			if len(line) < 2:
# 				continue

# 			l = line.split('\t')
# 			# [chapter, sent_num, tok_num, word, lemma, pos, syntax, [neg_trig, scope, event]{0,2} ]

# 			ch, sent_num = l[0], int(l[1])

# 			# Read parse tree data
# 			chapters[ch][sent_num][0].append(l[3])
# 			chapters[ch][sent_num][1].append(l[5])
# 			chapters[ch][sent_num][2] += l[6]
			
# 			# Read negation data
# 			if not '***' in l[-1]:
# 				n1 = [None if x == '_' else x for x in l[7:10]]
# 				chapters[ch][sent_num][3][0].append(n1)
# 				if len(l) > 10:
# 					n2 = [None if x == '_' else x for x in l[10:13]]
# 					chapters[ch][sent_num][3][1].append(n2)
				

# 	# Simplify chapter data

# 	# {chapter_name: [sent_vals]}
# 	chapter_sents = {name: [vals for _,vals in sorted(sents.items(), key=lambda x: x[0])] for name,sents in chapters.items()}

# 	# Construct constituent trees from input data, duplicating sentences if they have 2 negations

# 	# {chapter_name: [ParsedSentence]}
# 	chapter_trees = {}
# 	for name,corpus in chapter_sents.items():
# 		sents = []
# 		for vals in corpus:
# 			# 2 negations in this sentence
# 			if vals[3][1]:
# 				sents.append(ParsedSentence(*vals[:3], vals[3][0]))
# 				sents.append(ParsedSentence(*vals[:3], vals[3][1]))
# 			# 1 negation in this sentence
# 			elif vals[3][0]:
# 				sents.append(ParsedSentence(*vals[:3], vals[3][0]))
# 			# 0 negations in this sentence
# 			else:
# 				sents.append(ParsedSentence(*vals[:3]))
# 		chapter_trees[name] = sents

# 	return chapter_trees


def read_starsem_scope_data(fname, external_syntax_fname=None):
	''' Read in training data, parse sentence syntax, and construct consituent trees'''

	# Read in annotation data from corpus
	
	# sents : [[sent, pos, syntax, [neg_scope1, neg_scope2]]]
	sents = []
	new_sent = True

	with open(fname) as f:
		for line in f:
			if len(line) < 2:
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
				
	# Construct constituent trees from input data, duplicating sentences if they have 2 negations

	# sent_trees: [ParsedSentence]
	sent_trees = []

	# if using external syntax (ie CCG)
	if external_syntax_fname:
		external_syntax_trees = list(open(external_syntax_fname))

	for i,vals in enumerate(sents):
		syntax = external_syntax_trees[i] if external_syntax_fname else vals[2]
		syntax = syntax.replace('\n', '')

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

	return sent_trees


def _format_dataset(dataset, maxlen):
	# input: a corpus : [(A, ts, mask, cue, scope)]
	# output: reformatted for numpy : (A, ts, )
	dlen = len(dataset)
	Ashape = dataset[0][0].shape

	# A            = np.zeros((dlen, maxlen, maxlen))
	# adjust shape of A in case it's a directional matrix (will have extra indice)
	A            = np.zeros(tuple([dlen] + list(Ashape[0:(len(Ashape)-2)]) + [maxlen, maxlen]))
	ts           = np.zeros((dlen, maxlen), dtype=int)
	word_index   = np.zeros(dlen, dtype=object)
	cue          = np.zeros((dlen, maxlen))
	scope        = np.zeros(dlen, dtype=object)

	for i,d in enumerate(dataset):
		ds = d[0].shape
		
		if len(ds) == 3:
			A[i,:,0:ds[1],0:ds[2]] += d[0]
		else:
			A[i,0:ds[0],0:ds[1]] += d[0]
		ts[i,0:len(d[1])]      += np.array(d[1])
		word_index[i]           = np.array([i for i,v in enumerate(d[2]) if v])
		cue[i,0:len(d[3])]     += np.array(d[3])
		scope[i]                = np.array(d[4])

	return (A, ts, word_index, cue, scope)


# def format_data(train_unpacked, dev_unpacked, test_unpacked):
# 	# data format : [(A, ts, mask, cue, scope)]
# 	maxlen = max(max(len(x[1]) for x in d) for d in [train_unpacked, dev_unpacked, test_unpacked])
	
# 	train = _format_dataset(train_unpacked, maxlen)
# 	dev = _format_dataset(dev_unpacked, maxlen)
# 	test = _format_dataset(test_unpacked, maxlen)

# 	return train, dev, test

def format_data(corpora, word2ind):
	# input: a corpora : [[ParseTree]]
	# output: reformatted data of type : [(A, ts, mask, cue, scope)]
	data_splits = []
	for corpus in corpora:
		d = []
		for s in corpus:
			A, ts, word_mask = s.adjacency_matrix(directional=True)
			cue = [word2ind.get(w, UNK_TOK) for w in s.negation_cue()]
			scope = s.negation_surface_scope()
			d.append((A, ts, word_mask, cue, scope))
		data_splits.append(d)

	maxlen = max(max(len(s[1]) for s in d) for d in data_splits)
	return [_format_dataset(d, maxlen) for d in data_splits]

# def read_corpora(only_negations=False):
# 	corpora = []
# 	for corpus_file in [TRAINING_FILE, DEV_FILE, [TEST_FILE_A, TEST_FILE_B]]:
# 		chapters = {}
# 		if isinstance(corpus_file, list):
# 			for c in corpus_file:
# 				print('reading in:', c)
# 				chapters.update(read_starsem_scope_data(DATA_FOLDER + c))
# 		else:
# 			print('reading in:', corpus_file)
# 			chapters.update(read_starsem_scope_data(DATA_FOLDER + corpus_file))
		
# 		sents = [sent for _,chap in chapters.items() for sent in chap]
# 		if only_negations:
# 			sents = [s for s in sents if s.negation]
# 		corpora.append(sents)
# 	return corpora

def read_corpora(only_negations=False, external_syntax_folder=None):
	# output: list of corpora, each a list of ParseTree : [[ParseTree]]
	corpora = []
	for corpus_file in [TRAINING_FILE, DEV_FILE, [TEST_FILE_A, TEST_FILE_B]]:
		corpus = []
		if isinstance(corpus_file, list):
			for c in corpus_file:
				print('reading in:', c)
				ext_syntax = external_syntax_folder + c if external_syntax_folder else None
				corpus.append(read_starsem_scope_data(DATA_FOLDER + c, external_syntax_fname=ext_syntax))
		else:
			print('reading in:', corpus_file)
			ext_syntax = external_syntax_folder + corpus_file if external_syntax_folder else None
			corpus = read_starsem_scope_data(DATA_FOLDER + corpus_file, external_syntax_fname=ext_syntax)
		
		if only_negations:
			corpus = [s for s in corpus if s.negation]

		corpora.append(corpus)
		
	return corpora

def make_word_index(corpus):
	# input: a corpus, as a list of ParseTree : [ParseTree]
	# output: a mapping of unique words to unique integers, with additional UNK and PAD tokens
	all_cons = [(s.tree_node_tokens(), s.tree_leaf_tokens()) for s in corpus]
	sen_cons, sen_words = tuple(zip(*all_cons))
	cons_list, words_list = [x for s in sen_cons for x in s], [w for s in sen_words for w in s]
	cons, words = set(cons_list), set(words_list)
	vocab = cons | words

	word2ind = {w:i+2 for i,w in enumerate(vocab)}
	word2ind.update({'UNK':UNK_TOK, 'PAD':PAD_TOK})
	return word2ind

# def get_data(only_negations=False):
# 	# Read in corpus data
# 	corpora = read_corpora(only_negations=only_negations)

# 	# Build vocabulary from training data
# 	word2ind = make_word_index(corpora[0])

# 	return data_splits, word2ind

def get_parse_data(only_negations=False, external_syntax_folder=None):
	# Read in corpus data
	corpora = read_corpora(only_negations=only_negations, external_syntax_folder=external_syntax_folder)

	# Build vocabulary from training data
	word2ind = make_word_index(corpora[0])

	return corpora, word2ind





