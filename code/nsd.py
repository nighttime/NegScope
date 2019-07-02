from data import *
from nsdmodel import *
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
HIDDEN_UNITS = 100
GCN_LAYERS = 10
NUM_CLASSES = 2

EPOCHS = 12
LR = 0.005
BATCH_SIZE = 50

WEIGHT_DECAY = 5e-4
DROPOUT = 0.25


def build_model(vocab, directional=False):
	model = GCN(GCN_LAYERS, EMB_FEATURES, HIDDEN_UNITS, NUM_CLASSES, vocab, directional=directional)
	return model

def decode(inds, ind2word):
	return [ind2word[i] for i in inds]

def print_sent(tokens, word_index, ind2word, cue=None, scope=None):
	if isinstance(scope, torch.Tensor):
		scope = scope.data.numpy()

	s = '>> '
	for i,t in enumerate(tokens):
		begin = ''
		end = ''
		if i in word_index:
			if cue is not None and cue[i]:
				begin += Color.BOLD
				end += Color.ENDC
			if scope is not None and scope[word_index.tolist().index(i)]:
				begin += Color.UNDERLINE
				end += Color.ENDC
			s += begin + ind2word[t] + end + ' '
	print(s)

def print_batch(tokens, word_index, ind2word, cue=None, scope=None, cap=BATCH_SIZE):
	for b in range(tokens.shape[0]):
		if b < cap:
			print_sent(tokens[b], word_index[b], ind2word, cue[b] if cue is not None else None, scope[b] if scope is not None else None)


# def eval_batch(model, batch)

# def train_model(model, train, dev, ind2word):
# 	optimizer = optim.Adam(model.parameters(), lr=LR)#, weight_decay=WEIGHT_DECAY)
# 	A, ts, word_index, cue, scope = train
# 	datalen = A.shape[0]
# 	# class_wt = torch.tensor([0.1, 10.0]) # negated sentences to non-neg
# 	class_wt = torch.tensor([0.681,0.319]) # negated tokens to non-neg (given sentence has cue)
# 	f1_loss = F1_Loss()

# 	print('== BEGIN TRAINING ==')
# 	for epoch in range(EPOCHS):
# 		print('* Epoch: {:02d} -------------'.format(epoch))
# 		model.train()
		
# 		index = np.random.permutation(train[0].shape[0])
# 		ep_loss, ep_acc, ep_f1 = 0, 0, 0

# 		batch_inds = range(0, datalen, BATCH_SIZE)
# 		for i in batch_inds:
# 			batch_index_slice = slice(i,i+BATCH_SIZE)
# 			batch_slice = index[batch_index_slice]
# 			batch_len = batch_slice.shape[0]

# 			batch_A          = A[batch_slice]
# 			batch_ts         = ts[batch_slice]
# 			batch_word_index = word_index[batch_slice]
# 			batch_cue        = cue[batch_slice]
# 			batch_scope      = scope[batch_slice]

# 			optimizer.zero_grad()
# 			batch_output = model(batch_ts, batch_cue, batch_A)
# 			batch_actual = predict(batch_output)

# 			losses, accs, f1s = [], [], []
# 			for j in range(batch_len):
# 				class_output = batch_output[j,batch_word_index[j]]
# 				actual = batch_actual[j,batch_word_index[j]]
# 				expected = torch.tensor(batch_scope[j])
# 				logged_out = torch.log(class_output)
# 				losses.append(F.nll_loss(logged_out, expected, weight=class_wt))
# 				# losses.append(f1_loss(class_output[:,1], expected))
# 				accs.append(accuracy(actual, expected))
# 				f1s.append(f1_score(actual, expected))

# 			loss = sum(losses)/batch_len
# 			acc = sum(accs)/batch_len
# 			f1 = sum(f1s)/batch_len

# 			print_actual = lambda c=BATCH_SIZE: print_batch(batch_ts, batch_word_index, ind2word, cue=batch_cue, scope=batch_actual, cap=c)
# 			print_expected = lambda c=BATCH_SIZE: print_batch(batch_ts, batch_word_index, ind2word, cue=batch_cue, scope=batch_scope, cap=c)

# 			# if (i % 200 == 0 and epoch == 0) or i % 2000 == 0:
# 			if i % 150 == 0:
# 				print(' - Samples ({}/{})'.format(i, datalen), \
# 					'loss: {:0.4f}'.format(loss.item()), \
# 					'acc: {:0.4f}'.format(acc), \
# 					'f1: {:0.4f}'.format(f1))

# 				# if epoch == 2:
# 				# 	pdb.set_trace()

# 			ep_loss += loss.item()
# 			ep_acc += acc
# 			ep_f1 += f1
			
# 			# pdb.set_trace()

# 			loss.backward()
# 			optimizer.step()

# 		ep_loss /= len(batch_inds)
# 		ep_acc /= len(batch_inds)
# 		ep_f1 /= len(batch_inds)
# 		print(' :', \
# 			'loss_train: {:.4f} '.format(ep_loss), \
# 			'acc_train: {:.4f} '.format(ep_acc), \
# 			'f1_train: {:.4f} '.format(ep_f1))

# 		dev_loss, dev_acc, dev_f1 = eval_batch()
# 	print('FINISHED TRAINING')
# 	pdb.set_trace()

def train_model(model, train, dev, ind2word):
	optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
	A, ts, word_index, cue, scope = train
	datalen = A.shape[0]
	# class_wt = torch.tensor([0.1, 10.0]) # negated sentences to non-neg
	class_wt = torch.tensor([0.681,0.319]) # negated tokens to non-neg (given sentence has cue)
	f1_loss = F1_Loss()

	print('== BEGIN TRAINING ==')
	for epoch in range(EPOCHS):
		print('* Epoch: {:02d} -------------'.format(epoch))
		model.train()
		
		index = np.random.permutation(train[0].shape[0])
		ep_loss, ep_acc, ep_f1 = 0, 0, 0

		batch_inds = range(0, datalen, BATCH_SIZE)
		for i in batch_inds:
			batch_index_slice = slice(i,i+BATCH_SIZE)
			batch_slice = index[batch_index_slice]
			batch_len = batch_slice.shape[0]

			batch_A          = A[batch_slice]
			batch_ts         = ts[batch_slice]
			batch_word_index = word_index[batch_slice]
			batch_cue        = cue[batch_slice]
			batch_scope      = scope[batch_slice]

			optimizer.zero_grad()
			batch_output = model(batch_ts, batch_cue, batch_A)
			batch_actual = predict(batch_output)

			# Helper functions to visualize expected and actual outcomes
			print_actual = lambda c=BATCH_SIZE: print_batch(batch_ts, batch_word_index, ind2word, cue=batch_cue, scope=batch_actual, cap=c)
			print_expected = lambda c=BATCH_SIZE: print_batch(batch_ts, batch_word_index, ind2word, cue=batch_cue, scope=batch_scope, cap=c)

			losses, accs, f1s = [], [], []
			for j in range(batch_len):
				class_output = batch_output[j,batch_word_index[j]]
				actual = batch_actual[j,batch_word_index[j]]
				expected = torch.tensor(batch_scope[j])
				logged_out = torch.log(class_output)
				losses.append(F.nll_loss(logged_out, expected, weight=class_wt))
				# losses.append(f1_loss(class_output[:,1], expected))
				accs.append(accuracy(actual, expected))
				f1s.append(f1_score(actual, expected))
				# pdb.set_trace()

			loss = sum(losses)/batch_len
			acc = sum(accs)/batch_len
			f1 = sum(f1s)/batch_len

			if epoch == 5:
				pdb.set_trace()

			# if (i % 200 == 0 and epoch == 0) or i % 2000 == 0:
			if i % 150 == 0:
				print(' - Samples ({}/{})'.format(i, datalen), \
					'loss: {:0.4f}'.format(loss.item()), \
					'acc: {:0.4f}'.format(acc), \
					'f1: {:0.4f}'.format(f1))

			ep_loss += loss.item()
			ep_acc += acc
			ep_f1 += f1
			
			# pdb.set_trace()

			loss.backward()
			optimizer.step()

		ep_loss /= len(batch_inds)
		ep_acc /= len(batch_inds)
		ep_f1 /= len(batch_inds)
		print(' :', \
			'loss_train: {:.4f} '.format(ep_loss), \
			'acc_train: {:.4f} '.format(ep_acc), \
			'f1_train: {:.4f} '.format(ep_f1))

	print('FINISHED TRAINING')
	# pdb.set_trace()

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
    return F1.mean()

def test_model():
	pass
#	 model.eval()
#	 output = model(features, adj)
#	 loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#	 acc_test = accuracy(output[idx_test], labels[idx_test])
#	 print("Test set results:",
#		   "loss= {:.4f}".format(loss_test.item()),
#		   "accuracy= {:.4f}".format(acc_test.item()))


def main():
	# parser = argparse.ArgumentParser()
	# parser.add_argument('--syntax', type=str)
	# args = parser.parse_args()

	# if args.syntax and args.syntax[-1] != '/':
	# 	args.syntax += '/'

	syntax = '../starsem-st-2012-data/cd-sco-CCG/corpus/'
	directional = True

	# Retrieve data
	corpora, word2ind = get_parse_data(only_negations=True, external_syntax_folder=syntax)
	corpora[0] = [x for x in corpora[0] if x.longest_syntactic_path() <= GCN_LAYERS]
	print('GCN_LAYERS', GCN_LAYERS)
	print('num sents', len(corpora[0]))
	train, dev, test = format_data(corpora, word2ind, directional=directional)
	ind2word = {v:k for k,v in word2ind.items()}

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

	# pdb.set_trace()

	# Build model
	model = build_model(len(word2ind), directional=directional)

	# Train model
	train_model(model, train, dev, ind2word)

	exit(0)

	# Test model
	# test_model(model, test)


if __name__ == '__main__':
	main()
