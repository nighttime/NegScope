from data import *
from layers import *
# from layers import F1_Loss
from utils import *

import pdb
import time
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from pytorch_transformers import *


GCN  = 0
RNN  = 1
TRNN = 2

MODEL = RNN


if MODEL == RNN:
	EMB_FEATURES = 200
	POS_EMB_FEATURES = 32
	HIDDEN_UNITS = 200
	NUM_CLASSES = 2

	EPOCHS = 60
	LR = 0.0001
	BATCH_SIZE = 30

	WEIGHT_DECAY = 5e-4
	DROPOUT = 0.25

elif MODEL in (GCN, TRNN):
	EMB_FEATURES = 128
	POS_EMB_FEATURES = 16
	HIDDEN_UNITS = 200 #150
	GCN_LAYERS = 12
	NUM_CLASSES = 2

	EPOCHS = 50
	LR = 0.0001 #0.001 is good!
	BATCH_SIZE = 30

	WEIGHT_DECAY = 0
	DROPOUT_P = 0

LR_PATIENCE = 10
ES_PATIENCE = 2 * LR_PATIENCE



def build_gcn_model(vocab_size, directional=False):
	print(Color.BOLD + 'GCN MODEL' + Color.ENDC)
	model = GraphConvTagger(GCN_LAYERS, EMB_FEATURES, HIDDEN_UNITS, NUM_CLASSES, vocab_size, directional=directional)
	return model

def build_recurrent_model(vocab_size, pos_size):
	print(Color.BOLD + 'RNN MODEL' + Color.ENDC)
	model = RecurrentTagger(EMB_FEATURES, HIDDEN_UNITS, NUM_CLASSES, vocab_size, POS_EMB_FEATURES, pos_size)
	return model

def build_tree_recurrent_model(vocab_size, syntax_size, pretrained_embs):
	print(Color.BOLD + 'TRNN MODEL' + Color.ENDC)
	model = TreeRecurrentTagger(HIDDEN_UNITS, NUM_CLASSES, vocab_size, syntax_size, pretrained_embs, dropout_p=DROPOUT_P)
	return model

def decode(inds, ind2word):
	return [ind2word[i] for i in inds if ind2word[i] != 0]

def print_sent(tokens, word_index, ind2word, cue=None, scope=None, pos=None):
	if isinstance(scope, torch.Tensor):
		scope = scope.data.numpy()

	s = '>> '
	word_ct = 0
	for i,t in enumerate(tokens):
		if t == 0:
			break
		begin = ''
		end = ''
		# pdb.set_trace()
		if word_index is None or i in word_index:
			if cue is not None and cue[i] == 1:
				begin += Color.BOLD
				end += Color.ENDC
			if scope is not None:
				if ((word_index is not None and scope[word_index.tolist().index(i)]) or 
					(word_index is None and scope[i])):
					begin += Color.UNDERLINE
					end += Color.ENDC
			p = ''
			if pos is not None:
				p = '|' + ind2word[pos[word_ct]]
			s += begin + ind2word[t] + end + p + ' '
			word_ct += 1
	print(s)

def print_batch(tokens, ind2word, word_index=None, cue=None, scope=None, pos=None, cap=BATCH_SIZE):
	for b in range(tokens.shape[0]):
		if b < cap:
			print_sent(tokens[b], 
				word_index[b] if word_index is not None else None, 
				ind2word, 
				cue[b] if cue is not None else None, 
				scope[b] if scope is not None else None, 
				pos=(pos[0][b] if pos is not None else None)
				)


def run_model(model, train, dev, test, ind2word, ind2syn):
	optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=LR_PATIENCE, verbose=True)
	A, ts, pos, word_index, cue, scope, embs = train
	datalen = A.shape[0]
	f1_loss = F1_Loss()

	hist = {'train': [], 'dev': [], 'test': []}
	train_improvement_counter = 0
	test_improvement_counter = 0

	print('== BEGIN TRAINING ==')
	for epoch in range(EPOCHS):
		print(Color.BOLD + '* Epoch: {} -------------'.format(epoch) + Color.ENDC)
		model.train()
		
		index = np.random.permutation(datalen)
		ep_loss, ep_acc, ep_f1 = 0, 0, 0

		batch_inds = range(0, datalen, BATCH_SIZE)
		for i in batch_inds:
			batch_index_slice = slice(i,i+BATCH_SIZE)
			batch_slice = index[batch_index_slice]
			batch_len = batch_slice.shape[0]

			batch_A          = A[batch_slice]
			batch_ts         = ts[batch_slice]
			batch_pos        = pos[batch_slice]
			batch_word_index = word_index[batch_slice]
			batch_cue        = cue[batch_slice]
			batch_scope      = scope[batch_slice]
			if embs is not None:
				batch_embs   = embs[batch_slice]
			else:
				embs = None

			if MODEL == RNN:
				w_index = None
			elif MODEL in (GCN, TRNN):
				w_index = batch_word_index
			
			seq_lens = None

			# Pack data, removing tree structure
			if MODEL == RNN:
				seq_lens = pack_input_data_inplace(batch_ts, batch_cue, batch_word_index)
				
			# pdb.set_trace()
			optimizer.zero_grad()

			if MODEL == RNN:
				batch_output = model(batch_ts, batch_cue, batch_pos, batch_embs)
			elif MODEL == GCN:
				batch_output = model(batch_ts, batch_cue, batch_A)
			elif MODEL == TRNN:
				batch_output = model(batch_ts, batch_word_index, batch_cue, batch_A, batch_pos, batch_embs)
			
			loss, acc, f1, batch_actual = assess_batch(batch_output, batch_scope, batch_word_index, f1_loss, seq_lens)

			# Helper functions to visualize expected and actual outcomes
			print_actual = lambda c=BATCH_SIZE: print_batch(batch_ts, ind2word, word_index=w_index, cue=batch_cue, scope=batch_actual, cap=c)
			print_expected = lambda c=BATCH_SIZE: print_batch(batch_ts, ind2word, word_index=w_index, cue=batch_cue, scope=batch_scope, cap=c)
			print_actual_pos = lambda c=BATCH_SIZE: print_batch(batch_ts, ind2word, word_index=w_index, cue=batch_cue, scope=batch_scope, cap=c, pos=batch_pos)

			ep_loss += loss.item()
			ep_acc += acc
			ep_f1 += f1
			
			loss.backward()
			optimizer.step()

			progress = float(i+batch_len)/datalen
			print_progress(progress, info='{:.3f}'.format(f1))

		ep_loss /= len(batch_inds)
		ep_acc /= len(batch_inds)
		ep_f1 /= len(batch_inds)
		hist['train'].append([ep_loss, ep_acc, ep_f1])

		print()
		print_results(ep_loss, ep_acc, ep_f1, 'train', Color.OKBLUE)

		with torch.no_grad():
			# Dev set testing
			*dev_results, _ = test_dataset(model, dev, f1_loss)
			hist['dev'].append(dev_results)
			print_results(*dev_results, 'dev', Color.FAIL)

			# Test set testing
			*test_results, _ = test_dataset(model, test, f1_loss)
			hist['test'].append(test_results)
			print_results(*test_results, 'test', Color.OKGREEN)

		scheduler.step(dev_results[0])

		if epoch > 0 and test_results[2] <= hist['test'][-2][2]:
			test_improvement_counter += 1
		else:
			test_improvement_counter = 0
		if test_improvement_counter > ES_PATIENCE:
			print(Color.WARNING + '~~~ EARLY STOPPING ~~~' + Color.ENDC)
			break

	pdb.set_trace()
	print(Color.BOLD + '=== FINISHED ===' + Color.ENDC)

def pack_input_data_inplace(ts, cue, word_index):
	seq_lens = []
	for j in range(ts.shape[0]):
		new_ts  = ts[j,word_index[j]]
		new_cue = cue[j,word_index[j]]
		new_len = new_ts.shape[-1]
		seq_lens.append(new_len)
		ts[j].fill(0.)
		ts[j,:new_len] = new_ts
		cue[j].fill(0.)
		cue[j,:new_len] = new_cue
	return seq_lens

def test_dataset(model, dataset, loss_criterion):
	model.eval()
	# Unpack dev set data
	test_A, test_ts, test_pos, test_word_index, test_cue, test_scope, test_emb = dataset
	seq_lens = None

	# Pack data, removing tree structure
	if MODEL == RNN:
		seq_lens = pack_input_data_inplace(test_ts, test_cue, test_word_index)

	# Forward dataset through model
	if MODEL == GCN:
		test_output = model(test_ts, test_cue, test_A)
	elif MODEL == RNN:
		test_output = model(test_ts, test_cue, test_pos, test_emb)
	elif MODEL == TRNN:
		test_output = model(test_ts, test_word_index, test_cue, test_A, test_pos, test_emb)
	
	# Count metrics
	return assess_batch(test_output, test_scope, test_word_index, loss_criterion, seq_lens=seq_lens)

def assess_batch(model_output, y, word_index, loss_criterion, seq_lens=None):
	assert model_output.shape[0] == y.shape[0]
	batch_len = y.shape[0]

	yhat = predict(model_output)
	losses, accs, f1s = [], [], []

	# class_wt = torch.tensor([0.1, 10.0]) # negated sentences to non-neg
	class_wt = torch.tensor([0.681,0.319]) # negated tokens to non-neg (given sentence has cue)

	for j in range(batch_len):
		if MODEL == RNN:
			confidence_output = model_output[j,:seq_lens[j]]
			class_output = yhat[j,:seq_lens[j]]
			expected_output = torch.tensor(y[j])
		elif MODEL in (GCN, TRNN):
			confidence_output = model_output[j,word_index[j]]
			class_output = yhat[j,word_index[j]]
			expected_output = torch.tensor(y[j])

		# FOR Neg Log Likelihood
		# log_output = torch.log(confidence_output)
		# losses.append(F.nll_loss(log_output, expected_output, weight=class_wt))
		# OR FOR F1 LOSS
		losses.append(loss_criterion(confidence_output[:,1], expected_output))

		accs.append(accuracy(class_output, expected_output))
		f1s.append(f1_score(class_output, expected_output))

	loss = sum(losses)/batch_len
	acc = sum(accs)/batch_len
	f1 = sum(f[0] for f in f1s)/batch_len

	return loss, acc, f1, yhat

def print_results(loss, acc, f1, name, color=None):
	print((color + ' >' if color else ' -') + ' {:>6}'.format(name), \
		'loss: {:.4f} '.format(loss), \
		'acc: {:.4f} '.format(acc), \
		'f1: {:.4f} '.format(f1) + (Color.ENDC if color else ''))

def print_progress(progress, info='', bar_len=20):
	filled = int(progress*bar_len)
	print('\r[{}{}] {:.2f}% {}'.format('=' * filled, ' ' * (bar_len-filled), progress*100, info), end='')

def predict(class_output):
	return class_output.max(-1)[1]

def accuracy(actual, expected):
	correct = actual.eq(expected).double().sum()
	return correct / expected.shape[0]

def f1_score(actual, expected):
    actual = actual.double()
    expected = expected.double()

    tp = torch.sum(expected * actual, dim=-1)
    tn = torch.sum((1.0-expected) * (1.0-actual), dim=-1)
    fp = torch.sum((1.0-expected) * actual, dim=-1)
    fn = torch.sum(expected * (1 - actual), dim=-1)

    precision = tp / (tp + fp + EPSILON)
    recall    = tp / (tp + fn + EPSILON)

    F1 = 2 * precision * recall / (precision + recall + EPSILON)
    F1[torch.isnan(F1)] = 0.
    return F1.mean(), precision, recall


class EmbeddingModel:
	def __init__(self, model, tokenizer):
		self.model = model
		self.tokenizer = tokenizer

def main():
	# parser = argparse.ArgumentParser()
	# parser.add_argument('--syntax', type=str)
	# args = parser.parse_args()

	# if args.syntax and args.syntax[-1] != '/':
	# 	args.syntax += '/'

	syntax_folder = '../starsem-st-2012-data/cd-sco-CCG/corpus/'
	directional = True
	row_normalize = False
	pos_from_syntax=True
	condense_trees = True
	pretrained_embs = True

	# Retrieve data
	corpora, word2ind, syn2ind, full_vocab = get_parse_data(
		only_negations=True, 
		external_syntax_folder=syntax_folder,
		derive_pos_from_syntax=pos_from_syntax,
		condense_single_branches=condense_trees)

	ind2word = {v:k for k,v in word2ind.items()}
	ind2syn  = {v:k for k,v in syn2ind.items()}

	# corpora[0] = [x for x in corpora[0] if x.longest_syntactic_path() <= 7]
	# print('num sents', len(corpora[0]))
	a = [s for s in corpora[0] if any("'" in w for w in s.words if w != "''")]
	# pdb.set_trace()

	pre_embs = None
	if pretrained_embs:
		v = list(full_vocab)
		pretrained_weights = 'bert-base-uncased'
		bert_model = BertModel.from_pretrained(pretrained_weights, output_hidden_states=True)
		# bert_model = BertForTokenClassification.from_pretrained(pretrained_weights, output_hidden_states=True)
		bert_tokenizer = BertTokenizer.from_pretrained(pretrained_weights, never_split=v, do_basic_tokenize=False)
		pre_embs = EmbeddingModel(bert_model, bert_tokenizer)
		# pdb.set_trace()

	train, dev, test = format_data(
		corpora, 
		word2ind, 
		syn2ind, 
		directional=directional, 
		row_normalize=row_normalize, 
		pretrained_embs_model=pre_embs)
	

	# for c in corpora:
	# 	lengths = [t.longest_syntactic_path() for t in c]
	# 	print('CORPUS', len(lengths))
	# 	print(' 0-5 ', len([x for x in lengths if  0 < x <= 5]))
	# 	print(' 5-10', len([x for x in lengths if  5 < x <= 10]))
	# 	print('10-15', len([x for x in lengths if 10 < x <= 15]))
	# 	print('15-20', len([x for x in lengths if 15 < x <= 20]))
	# 	print('20-25', len([x for x in lengths if 20 < x <= 25]))
	# 	print('25-30', len([x for x in lengths if 25 < x <= 30]))
	# 	print('30-45', len([x for x in lengths if 30 < x <= 45]))

	

	#	A, ts, word_index, cue, scope = train
	# all_toks = 0
	# neg_toks = 0
	# for i in range(train[1].shape[0]):
	# 	all_toks += len(train[2][i])
	# 	neg_toks += np.sum(train[4][i])
	# print(neg_toks/float(all_toks),neg_toks,all_toks)

	# Build model
	if MODEL == GCN:
		model = build_gcn_model(len(word2ind), directional=directional)
	elif MODEL == RNN:
		model = build_recurrent_model(len(word2ind), len(syn2ind))
	elif MODEL == TRNN:
		model = build_tree_recurrent_model(len(word2ind), len(syn2ind), pretrained_embs)
	# pdb.set_trace()
	# Train and Test model
	run_model(model, train, dev, test, ind2word, ind2syn)



if __name__ == '__main__':
	main()
