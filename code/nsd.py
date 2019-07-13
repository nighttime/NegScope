from data import *
from layers import *
from utils import *
import pdb
import time
import argparse
# import torchviz

import torch
import torch.nn.functional as F
import torch.optim as optim
from layers import F1_Loss

EMB_FEATURES = 128
POS_EMB_FEATURES = 16
HIDDEN_UNITS = 200
GCN_LAYERS = 12
NUM_CLASSES = 2

EPOCHS = 10
LR = 0.01
BATCH_SIZE = 25

WEIGHT_DECAY = 5e-4
DROPOUT = 0.25


def build_gcn_model(vocab_size, directional=False):
	model = GCN(GCN_LAYERS, EMB_FEATURES, HIDDEN_UNITS, NUM_CLASSES, vocab_size, directional=directional)
	return model

def build_recurrent_model(vocab_size, pos_size):
	model = RecurrentTagger(EMB_FEATURES, HIDDEN_UNITS, NUM_CLASSES, vocab_size, POS_EMB_FEATURES, pos_size)
	return model

def build_tree_recurrent_model(vocab_size):
	model = TreeRecurrentTagger(HIDDEN_UNITS, NUM_CLASSES, vocab_size)
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
				p = '|' + pos[1][pos[0][word_ct]]
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
				pos=((pos[0][b], pos[1]) if pos is not None else None)
				)

def assess_batch(model_output, y, word_index, loss_criterion):
	assert model_output.shape[0] == y.shape[0]
	batch_len = y.shape[0]

	yhat = predict(model_output)
	losses, accs, f1s = [], [], []

	for j in range(batch_len):
		# FOR BiLSTM ######
		# confidence_output = model_output[j,:seq_lens[j]]
		# class_output = yhat[j,:seq_lens[j]]
		# expected_output = torch.tensor(y[j])
		# FOR GCN & TRNN####
		confidence_output = model_output[j,word_index[j]]
		class_output = yhat[j,word_index[j]]
		expected_output = torch.tensor(y[j])
		###################
		

		# FOR Neg Log Likelihood
		# logged_out = torch.log(confidence_output)
		# losses.append(F.nll_loss(logged_out, expected_output))#, weight=class_wt))

		# OR FOR F1 LOSS
		losses.append(loss_criterion(confidence_output[:,1], expected_output))

		accs.append(accuracy(class_output, expected_output))
		f1s.append(f1_score(class_output, expected_output))

	loss = sum(losses)/batch_len
	acc = sum(accs)/batch_len
	f1 = sum(f[0] for f in f1s)/batch_len

	return loss, acc, f1



def run_model(model, train, dev, test, ind2word, ind2pos):
	optimizer = optim.Adam(model.parameters(), lr=LR)#, weight_decay=WEIGHT_DECAY)
	A, ts, pos, word_index, cue, scope = train
	datalen = A.shape[0]
	# class_wt = torch.tensor([0.1, 10.0]) # negated sentences to non-neg
	class_wt = torch.tensor([0.681,0.319]) # negated tokens to non-neg (given sentence has cue)
	f1_loss = F1_Loss()

	print('== BEGIN TRAINING ==')
	for epoch in range(EPOCHS):
		print(Color.BOLD + '* Epoch: {:02d} -------------'.format(epoch) + Color.ENDC)
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

			# Helper functions to visualize expected and actual outcomes
			# FOR biLSTM ######
			# w_index = None
			# ELSE FOR GCN & TRNN ####
			w_index = batch_word_index
			###################
			print_actual = lambda c=BATCH_SIZE: print_batch(batch_ts, ind2word, word_index=w_index, cue=batch_cue, scope=batch_actual, cap=c)
			print_expected = lambda c=BATCH_SIZE: print_batch(batch_ts, ind2word, word_index=w_index, cue=batch_cue, scope=batch_scope, cap=c)
			print_actual_pos = lambda c=BATCH_SIZE: print_batch(batch_ts, ind2word, word_index=w_index, cue=batch_cue, scope=batch_scope, cap=c, pos=(batch_pos, ind2pos))
			
			# FOR biLSTM ######
			# seq_lens = []
			# for j in range(batch_len):
			# 	new_ts  = batch_ts[j,batch_word_index[j]]
			# 	new_cue = batch_cue[j,batch_word_index[j]]
			# 	new_len = new_ts.shape[-1]
			# 	seq_lens.append(new_len)
			# 	batch_ts[j].fill(0.)
			# 	batch_ts[j,:new_len] = new_ts
			# 	batch_cue[j].fill(0.)
			# 	batch_cue[j,:new_len] = new_cue
			###################

			optimizer.zero_grad()
			# FOR BiLSTM ######
			# batch_output = model(batch_ts, batch_pos, batch_cue)
			# FOR GCN ####
			# batch_output = model(batch_ts, batch_cue, batch_A)
			# FOR TRNN ####
			batch_output = model(batch_ts, batch_word_index, batch_cue, batch_A)
			###################
			
			# batch_actual = predict(batch_output)
			# losses, accs, f1s = [], [], []

			# for j in range(batch_len):
			# 	# FOR BiLSTM ######
			# 	# class_output = batch_output[j,:seq_lens[j]]
			# 	# actual = batch_actual[j,:seq_lens[j]]
			# 	# expected = torch.tensor(batch_scope[j])
			# 	# FOR GCN & TRNN####
			# 	class_output = batch_output[j,batch_word_index[j]]
			# 	actual = batch_actual[j,batch_word_index[j]]
			# 	expected = torch.tensor(batch_scope[j])
			# 	###################
				

			# 	# FOR Neg Log Likelihood
			# 	# logged_out = torch.log(class_output)
			# 	# losses.append(F.nll_loss(logged_out, expected))#, weight=class_wt))

			# 	# OR FOR F1 LOSS
			# 	losses.append(f1_loss(class_output[:,1], expected))

			# 	accs.append(accuracy(actual, expected))
			# 	f1s.append(f1_score(actual, expected))

			# loss = sum(losses)/batch_len
			# acc = sum(accs)/batch_len
			# f1 = sum(f[0] for f in f1s)/batch_len

			loss, acc, f1 = assess_batch(batch_output, batch_scope, batch_word_index, f1_loss)

			if i % 400 == 0:
				print(' - Samples {}-{} (of {})'.format(i, i+batch_len, datalen), \
					'loss: {:0.4f}'.format(loss.item()), \
					'acc: {:0.4f}'.format(acc), \
					'f1: {:0.4f}'.format(f1))

			ep_loss += loss.item()
			ep_acc += acc
			ep_f1 += f1
			
			loss.backward()
			optimizer.step()

		ep_loss /= len(batch_inds)
		ep_acc /= len(batch_inds)
		ep_f1 /= len(batch_inds)
		print(Color.OKBLUE + ' > train', \
			'loss: {:.4f} '.format(ep_loss), \
			'acc: {:.4f} '.format(ep_acc), \
			'f1: {:.4f} '.format(ep_f1),
			Color.ENDC)

		# Dev set testing
		# Unpack dev set data
		dev_A, dev_ts, dev_pos, dev_word_index, dev_cue, dev_scope = dev
		# Forward batch through model
		dev_output = model(dev_ts, dev_word_index, dev_cue, dev_A)
		# Count metrics and print
		dev_loss, dev_acc, dev_f1 = assess_batch(dev_output, dev_scope, dev_word_index, f1_loss)
		print(Color.OKGREEN + ' >   dev', \
			'loss: {:.4f} '.format(dev_loss), \
			'acc: {:.4f} '.format(dev_acc), \
			'f1: {:.4f} '.format(dev_f1),
			Color.ENDC)

	print(Color.BOLD + '=== FINISHED TRAINING ===' + Color.ENDC)
	# pdb.set_trace()

	# Test Model
	model.eval()
	print(Color.BOLD + '=== BEGIN TESTING ===' + Color.ENDC)
	# Unpack dev set data
	test_A, test_ts, test_pos, test_word_index, test_cue, test_scope = test
	# Forward dataset through model
	test_output = model(test_ts, test_word_index, test_cue, test_A)
	# Count metrics and print
	test_loss, test_acc, test_f1 = assess_batch(test_output, test_scope, test_word_index, f1_loss)
	print(Color.WARNING + ' >  test', \
		'loss: {:.4f} '.format(test_loss), \
		'acc: {:.4f} '.format(test_acc), \
		'f1: {:.4f} '.format(test_f1),
		Color.ENDC)

	print(Color.BOLD + '=== FINISHED TESTING ===' + Color.ENDC)

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


def main():
	# parser = argparse.ArgumentParser()
	# parser.add_argument('--syntax', type=str)
	# args = parser.parse_args()

	# if args.syntax and args.syntax[-1] != '/':
	# 	args.syntax += '/'

	syntax = '../starsem-st-2012-data/cd-sco-CCG/corpus/'
	directional = True
	row_normalize = False
	condense = True

	# Retrieve data
	corpora, word2ind, pos2ind = get_parse_data(
		only_negations=True, 
		external_syntax_folder=syntax,
		condense_single_branches=condense)
	# corpora[0] = [x for x in corpora[0] if x.longest_syntactic_path() <= 7]
	# print('GCN_LAYERS', GCN_LAYERS)

	# for t in corpora[0]:
	# 	cs = [c.constituent for c in t.constituents]
	# 	fcs = [c for c in t.constituents if len(c.children)==1 and not c.children[0].is_leaf()]
	# 	if fcs:
	# 		pdb.set_trace()

	print('num sents', len(corpora[0]))
	train, dev, test = format_data(corpora, word2ind, pos2ind, directional=directional, row_normalize=row_normalize)
	ind2word = {v:k for k,v in word2ind.items()}
	ind2pos = {v:k for k,v in pos2ind.items()}

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

	

	# pdb.set_trace()

	#	A, ts, word_index, cue, scope = train
	# all_toks = 0
	# neg_toks = 0
	# for i in range(train[1].shape[0]):
	# 	all_toks += len(train[2][i])
	# 	neg_toks += np.sum(train[4][i])

	# print(neg_toks/float(all_toks),neg_toks,all_toks)

	# Build model
	# model = build_gcn_model(len(word2ind), directional=directional)
	# model = build_word_model(len(word2ind), len(pos2ind))
	model = build_tree_recurrent_model(len(word2ind))

	# Train and Test model
	run_model(model, train, dev, test, ind2word, ind2pos)



if __name__ == '__main__':
	main()
