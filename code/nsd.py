from data import *
from nsdmodel import *
from utils import *
import pdb
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

EMB_FEATURES = 32
HIDDEN_UNITS = 32
GCN_LAYERS = 20
NUM_CLASSES = 2

EPOCHS = 6
LR = 0.05
BATCH_SIZE = 100

WEIGHT_DECAY = 5e-4
DROPOUT = 0.25


def build_model(vocab):
	model = GCN(GCN_LAYERS, EMB_FEATURES, HIDDEN_UNITS, NUM_CLASSES, vocab)
	return model

def decode(inds, ind2word):
	return [ind2word[i] for i in inds]

def print_sent(tokens, word_index, cue, scope, ind2word):
	s = '>> '
	for i,t in enumerate(tokens):
		begin = ''
		end = ''
		if i in word_index:
			if cue[i]:
				begin += Color.BOLD
				end += Color.ENDC
			if scope[word_index.tolist().index(i)]:
				begin += Color.UNDERLINE
				end += Color.ENDC
			s += begin + ind2word[t] + end + ' '
	print(s)

def print_batch(tokens, word_index, cue, scope, ind2word):
	for b in range(tokens.shape[0]):
		print_sent(tokens[b], word_index[b], cue[b], scope[b], ind2word)


def train_model(model, train, dev, ind2word):
	optimizer = optim.Adam(model.parameters(), lr=LR)#, weight_decay=WEIGHT_DECAY)
	A, ts, word_index, cue, scope = train
	datalen = A.shape[0]
	t = time.time()
	
	print('== BEGIN TRAINING ==')
	for epoch in range(EPOCHS):
		model.train()
		
		index = np.random.permutation(train[0].shape[0])
		ep_loss, ep_acc = 0, 0

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
			output = model(batch_ts, batch_cue, batch_A)
			# pdb.set_trace()

			class_wt = torch.tensor([0.85, 0.15])
			losses, accs = [], []
			for j in range(batch_len):
				actual = output[j,batch_word_index[j]]
				expected = torch.tensor(batch_scope[j])
				losses.append(F.nll_loss(actual, expected, weight=class_wt))
				accs.append(accuracy(actual.max(1)[1], expected))

			loss = sum(losses)/batch_len
			acc = sum(accs)/batch_len
			
			if i % 500 == 0:
				print(' - Samples ({}/{})'.format(i, datalen), \
					'loss: {:0.4f}'.format(loss.item()), \
					'acc: {:0.4f}'.format(acc))

			ep_loss += loss.item()
			ep_acc += acc
			
			# pdb.set_trace()

			loss.backward()
			optimizer.step()

		ep_loss /= len(batch_inds)
		ep_acc /= len(batch_inds)
		print('| Epoch: {:04d}'.format(epoch), \
			'loss_train: {:.4f}'.format(ep_loss), \
			'acc_train: {:.4f}'.format(ep_acc), \
			'time: {:.4f}s'.format(time.time() - t))
		t = time.time()

def accuracy(actual, expected):
	correct = actual.eq(expected).double().sum()
	return correct / expected.shape[0]

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
	# Retrieve data
	(train, dev, test), word2ind = get_data()

	ind2word = {v:k for k,v in word2ind.items()}

	train, dev, test = format_data(train, dev, test)

	# pdb.set_trace()

	# Build model
	model = build_model(len(word2ind))

	# Train model
	train_model(model, train, dev, ind2word)

	exit(0)

	# Test model
	test_model(model, test)


if __name__ == '__main__':
	main()
