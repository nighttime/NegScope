from data import *
from nsdmodel import *
from utils import *
import pdb
import time
# import torchviz

import torch
import torch.nn.functional as F
import torch.optim as optim
from layers import F1_Loss

EMB_FEATURES = 128
HIDDEN_UNITS = 100
GCN_LAYERS = 10
NUM_CLASSES = 2

EPOCHS = 5
LR = 0.01
BATCH_SIZE = 25

WEIGHT_DECAY = 5e-4
DROPOUT = 0.25


def build_model(vocab):
	model = GCN(GCN_LAYERS, EMB_FEATURES, HIDDEN_UNITS, NUM_CLASSES, vocab)
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


def train_model(model, train, dev, ind2word):
	optimizer = optim.Adam(model.parameters(), lr=LR)#, weight_decay=WEIGHT_DECAY)
	A, ts, word_index, cue, scope = train
	datalen = A.shape[0]
	class_wt = torch.tensor([0.1, 10.0])
	f1_loss = F1_Loss()
	t = time.time()

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

			losses, accs, f1s = [], [], []
			for j in range(batch_len):
				class_output = batch_output[j,batch_word_index[j]]
				actual = batch_actual[j,batch_word_index[j]]
				expected = torch.tensor(batch_scope[j])
				logged_out = torch.log(class_output)
				losses.append(F.nll_loss(logged_out, expected))#, weight=class_wt))
				# losses.append(f1_loss(class_output[:,1], expected))
				accs.append(accuracy(actual, expected))
				f1s.append(f1_score(actual, expected))

			loss = sum(losses)/batch_len
			acc = sum(accs)/batch_len
			f1 = sum(f1s)/batch_len

			print_actual = lambda c=BATCH_SIZE: print_batch(batch_ts, batch_word_index, ind2word, cue=batch_cue, scope=batch_actual, cap=c)
			print_expected = lambda c=BATCH_SIZE: print_batch(batch_ts, batch_word_index, ind2word, cue=batch_cue, scope=batch_scope, cap=c)

			# if (i % 200 == 0 and epoch == 0) or i % 2000 == 0:
			if i % 150 == 0:
				print(' - Samples ({}/{})'.format(i, datalen), \
					'loss: {:0.4f}'.format(loss.item()), \
					'acc: {:0.4f}'.format(acc), \
					'f1: {:0.4f}'.format(f1))

				# if epoch == 2:
				# 	pdb.set_trace()

			ep_loss += loss.item()
			ep_acc += acc
			ep_f1 += f1
			
			# pdb.set_trace()

			# torchviz.make_dot(loss)

			loss.backward()
			# pdb.set_trace()
			optimizer.step()

		ep_loss /= len(batch_inds)
		ep_acc /= len(batch_inds)
		ep_f1 /= len(batch_inds)
		print(' :', \
			'loss_train: {:.4f} '.format(ep_loss), \
			'acc_train: {:.4f} '.format(ep_acc), \
			'f1_train: {:.4f} '.format(ep_f1))
			# 'time: {:.4f}s'.format(time.time() - t))
		t = time.time()
	print('FINISHED TRAINING')
	pdb.set_trace()

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
	# Retrieve data
	(train, dev, test), word2ind = get_data(only_negations=True)

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
