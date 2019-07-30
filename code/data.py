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

	sentence_copies = []

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
			sentence_copies.extend([1,2])
		elif vals[3][0]:
		# 1 negation in this sentence
			sent_trees.append(ParsedSentence(*vals[:2], syntax, negation=vals[3][0]))
			sentence_copies.append(1)
		else:
		# 0 negations in this sentence
			sent_trees.append(ParsedSentence(*vals[:2], syntax))
			sentence_copies.append(1)

		ext_p += 1

	return sent_trees, sentence_copies

def _prepare_starsem_output_scope_data(gold_lines, model_outputs, corpus_data, sent_ptr=0):
	A, ts, pos, word_index, cue, scope, embs = corpus_data
	sents_out = []
	gold_word_ptr = 0
	sent_ptr_incr = 0
	for i,line in enumerate(gold_lines):

		if line == '\n':
			# if i > 0:
			# 	try:
			# 		assert int(gold_lines[i-1].split('\t')[2]+1) == len(word_index[sent_ptr])
			# 	except:
			# 		pdb.set_trace()
			sents_out.append('')
			sent_ptr += sent_ptr_incr
			sent_ptr_incr = 0
			gold_word_ptr = 0
			double_neg = False
			continue

		line_bits = line.strip().split('\t')
		
		# Negation in this sentence
		if line_bits[7] != '***':
			sent_ptr_incr = 1
			words = word_index[sent_ptr]
			in_scope = model_outputs[sent_ptr][words][gold_word_ptr]
			if in_scope:
				base_word = line_bits[3]
				sent_cue = line_bits[7]
				# In the case where a cue is only part of a word, 
				if sent_cue != '_':
					scope_word = base_word.replace(sent_cue, '', 1)
					line_bits[8] = scope_word
				else:
					line_bits[8] = base_word
			else:
				line_bits[8] = '_'
			line_bits[9] = '_'

			# Second negation in this sentence
			if len(line_bits) > 10:
				sent_ptr_incr = 2
				sent_ptr2 = sent_ptr+1
				# try:
				assert len(word_index[sent_ptr]) == len(word_index[sent_ptr2])
				# except:
				# 	pdb.set_trace()
				words = word_index[sent_ptr2]
				in_scope = model_outputs[sent_ptr2][words][gold_word_ptr]
				if in_scope:
					base_word = line_bits[3]
					sent_cue = line_bits[10]
					# In the case where a cue is only part of a word, 
					if sent_cue != '_':
						scope_word = base_word.replace(sent_cue, '', 1)
						line_bits[11] = scope_word
					else:
						line_bits[11] = base_word
				else:
					line_bits[11] = '_'
				line_bits[12] = '_'

		sents_out.append(line_bits)
		gold_word_ptr += 1

	return sents_out, sent_ptr

def write_starsem_scope_data(out_fname1, out_fname2, model_outputs, corpus_data):
	sent_ptr = 0

	cardboard = DATA_FOLDER + TEST_FILE_A
	circle    = DATA_FOLDER + TEST_FILE_B

	with open(cardboard, 'r') as gold_A:
		lines1 = gold_A.readlines()

	with open(circle, 'r') as gold_B:
		lines2 = gold_B.readlines()
	
	sents_out1, sent_ptr = _prepare_starsem_output_scope_data(lines1, model_outputs, corpus_data)
	sents_out2, _        = _prepare_starsem_output_scope_data(lines2, model_outputs, corpus_data, sent_ptr=sent_ptr)

	with open(out_fname1, 'w') as outfile:
		for line in sents_out1:
			outfile.write('\t'.join(line) + '\n')

	with open(out_fname2, 'w') as outfile:
		for line in sents_out2:
			outfile.write('\t'.join(line) + '\n')



def _format_dataset(dataset, maxlen, pre_embs_path=None):
	# input: a corpus : [(A, ts, mask, cue, scope)]
	# output: reformatted as numpy arrays : (A, ts, mask, cue, scope)
	dlen = len(dataset)
	Ashape = dataset[0][0].shape
	# pretrained_embs = len(dataset[0]) > 6

	# A            = np.zeros((dlen, maxlen, maxlen))
	# adjust shape of A in case it's a directional matrix (will have extra indice)
	A            = np.zeros(tuple([dlen] + list(Ashape[0:(len(Ashape)-2)]) + [maxlen, maxlen]))
	ts           = np.zeros((dlen, maxlen), dtype=int)
	pos          = np.zeros((dlen, maxlen), dtype=int)
	word_index   = np.zeros(dlen, dtype=object)
	cue          = np.zeros((dlen, maxlen))
	scope        = np.zeros(dlen, dtype=object)
	# if pretrained_embs:
	# 	embs     = np.zeros(dlen, dtype=object)
	# else:
	# 	embs     = None
	if pre_embs_path:
		embs     = np.load(pre_embs_path, allow_pickle=True)
	else:
		embs     = None

	for i,d in enumerate(dataset):
		ds = d[0].shape
		
		if len(ds) == 3:
			A[i,:,0:ds[1],0:ds[2]] += d[0]
		else:
			A[i,0:ds[0],0:ds[1]] += d[0]
		ts[i,0:len(d[1])]      += np.array(d[1])
		pos[i,0:len(d[2])]     += np.array(d[2])
		word_index[i]           = np.array([i for i,v in enumerate(d[3]) if v])
		cue[i,0:len(d[3])]     += np.array(d[4])
		scope[i]                = np.array(d[5])

		# if pretrained_embs:
		# 	embs[i]             = d[6]

	return (A, ts, pos, word_index, cue, scope, embs)


def format_data(corpora, word2ind, syn2ind, directional=False, row_normalize=True, embs_folder=None, pretrained_embs_model=None):
	# input: a corpora : [[ParseTree]]
	# output: reformatted data of type : [(A, ts, mask, cue, scope)]
	assert len(corpora) == 3
	data_splits = []
	for corpus in corpora:
		d = []
		for i,s in enumerate(corpus):
			A, toks, word_mask = s.adjacency_matrix(directional=directional, row_normalize=row_normalize)
			ts = [word2ind.get(t, UNK_TOK) if word_mask[i] else syn2ind.get(t, UNK_TOK) for i,t in enumerate(toks)]
			pos = [syn2ind.get(p, UNK_TOK) for p in s.pos]
			cue = s.negation_cue()
			scope = s.negation_surface_scope()
			items = [A, ts, pos, word_mask, cue, scope]

			# if pretrained_embs_model is not None:
			# 	words = s.tree_leaf_tokens()
			# 	sent = ' '.join(words)
			# 	tokens = pretrained_embs_model.tokenizer.encode(sent)
			# 	input_ids = torch.tensor([tokens])
			# 	model_outputs = pretrained_embs_model.model(input_ids)
			# 	hidden_states = model_outputs[0].squeeze() # model outputs
			# 	# hidden_states = model_outputs[2][4].squeeze() # model intermediate layer outputs
			# 	partial_toks = [pretrained_embs_model.tokenizer._convert_id_to_token(tokens[i]) for i in range(len(tokens))]
			# 	embs = np.zeros([len(words), hidden_states.shape[-1]])
				
			# 	ct = -1
			# 	for j,t in enumerate(partial_toks):
			# 		if not t.startswith('##'):
			# 			ct += 1
			# 		embs[ct] += hidden_states[j].detach().numpy()
			# 	items.append(embs)

			d.append(tuple(items))
			print('\rencoding dataset: {:.3f}'.format(float(i+1)/len(corpus)), end='')
		print()

		data_splits.append(d)

	maxlen = max(max(len(s[1]) for s in d) for d in data_splits)
	embs_fnames = ['train.npy', 'dev.npy', 'test.npy']
	return [_format_dataset(d, maxlen, pre_embs_path=(embs_folder + '/' + embs_fnames[i] if embs_folder else None)) for i,d in enumerate(data_splits)]
	# return [_format_dataset(d, maxlen) for i,d in enumerate(data_splits)]

def read_corpora(only_words=False, only_negations=False, external_syntax_folder=None, derive_pos_from_syntax=False, condense_single_branches=False):
	# output: list of corpora, each a list of ParseTree : [[ParseTree]] OR a list of words : [[str]]
	corpora = []
	corp_sent_copies = []
	for corpus_file in [TRAINING_FILE, DEV_FILE, [TEST_FILE_A, TEST_FILE_B]]:
		corpus = []
		sent_copies = []
		if isinstance(corpus_file, list):
			for c in corpus_file:
				print('reading in:', c)
				ext_syntax = external_syntax_folder + c if external_syntax_folder else None
				if only_words:
					corpus += _starsem_sentence_vals(DATA_FOLDER + c)
				else:
					corpus_, sent_copies_ = read_starsem_scope_data(DATA_FOLDER + c, external_syntax_fname=ext_syntax)
					corpus += corpus_
					sent_copies += sent_copies_
		else:
			print('reading in:', corpus_file)
			ext_syntax = external_syntax_folder + corpus_file if external_syntax_folder else None
			if only_words:
				corpus = _starsem_sentence_vals(DATA_FOLDER + corpus_file)
			else:
				corpus_, sent_copies_ = read_starsem_scope_data(DATA_FOLDER + corpus_file, external_syntax_fname=ext_syntax)
				corpus += corpus_
				sent_copies += sent_copies_
		
		if only_negations:
			if only_words:
				# Keep sentence values if the negation component contains a scope
				corpus = [vs for vs in corpus if len(vs[-1][0])]
			else:
				corpus, sent_copies = zip(*[s for s in zip(corpus, sent_copies) if s[0].negation])
				corpus, sent_copies = list(corpus), list(sent_copies)

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
		corp_sent_copies.append(sent_copies)

	return corpora, corp_sent_copies

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


def get_parse_data(
	only_negations=False, 
	external_syntax_folder=None, 
	derive_pos_from_syntax=False, 
	condense_single_branches=False):
	# Read in corpus data
	corpora, corp_sent_copies = read_corpora(
		only_negations=only_negations, 
		external_syntax_folder=external_syntax_folder, 
		derive_pos_from_syntax=derive_pos_from_syntax,
		condense_single_branches=condense_single_branches)

	# Build vocabulary from training data
	word2ind, syn2ind = make_corpus_index(corpora[0])

	train_voc = set(word2ind.keys())
	dev_voc   = set(make_corpus_index(corpora[1])[0].keys())
	test_voc  = set(make_corpus_index(corpora[2])[0].keys())

	full_vocab = train_voc | dev_voc | test_voc

	return corpora, corp_sent_copies, word2ind, syn2ind, full_vocab





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








