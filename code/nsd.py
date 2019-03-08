from data import *
from nsdmodel import *
import pdb
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

DATA_FOLDER = '../starsem-st-2012-data/cd-sco/corpus/'
TRAINING_FILE = 'training/SEM-2012-SharedTask-CD-SCO-training-09032012.txt'
DEV_FILE = 'dev/SEM-2012-SharedTask-CD-SCO-dev-09032012.txt'
TEST_FILE_A = 'test-gold/SEM-2012-SharedTask-CD-SCO-test-cardboard-GOLD.txt'
TEST_FILE_B = 'test-gold/SEM-2012-SharedTask-CD-SCO-test-circle-GOLD.txt'

EPOCHS = 10
LR = 0.01
BATCH_SIZE = 15
WEIGHT_DECAY = 5e-4
CONSTITUENT_FEATURES = 128
HIDDEN_UNITS = 64
NUM_CLASSES = 2
DROPOUT = 0.25

UNK_TOK = 0

def build_model(vocab):
	model = GCN(3, CONSTITUENT_FEATURES, HIDDEN_UNITS, NUM_CLASSES, vocab)
	return model

def train_model(model, train, dev):
	model.train()
	optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
	t = time.time()
	print('== BEGIN TRAINING ==')
	for epoch in range(EPOCHS):
		shuffle(train)
		for sample in train:
			optimizer.zero_grad()

			A, ts, mask, cue, scope = sample

			output = model(ts, cue, A)
			# pdb.set_trace()
			inds = [i for i,v in enumerate(mask) if v]
			loss = F.nll_loss(output[inds], torch.tensor(scope))
			# acc_train = accuracy(output[idx_train], labels[idx_train])
			loss.backward()
			optimizer.step()

			# if not args.fastmode:
				# Evaluate validation set performance separately,
				# deactivates dropout during validation run.
				# model.eval()
				# output = model(features, adj)

			loss_val = F.nll_loss(output[inds], torch.tensor(scope))
			# acc_val = accuracy(output[idx_val], labels[idx_val])
			print('Epoch: {:04d}'.format(epoch+1),
				  'loss_train: {:.4f}'.format(loss.item()),
				  'loss_val: {:.4f}'.format(loss_val.item()),
				  'time: {:.4f}s'.format(time.time() - t))


def test_model():
	pass
#	 model.eval()
#	 output = model(features, adj)
#	 loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#	 acc_test = accuracy(output[idx_test], labels[idx_test])
#	 print("Test set results:",
#		   "loss= {:.4f}".format(loss_test.item()),
#		   "accuracy= {:.4f}".format(acc_test.item()))



def get_data():

	# Read in corpus data
	corpora = []
	for corpus_file in [TRAINING_FILE, DEV_FILE, TEST_FILE_A]:
		print('reading in:', corpus_file)
		chapters = read_starsem_scope_data(DATA_FOLDER + corpus_file)
		sents = [sent for _,chap in chapters.items() for sent in chap]
		corpora.append(sents)

	# Build vocabulary from training data
	all_cons = [(s.tree_node_tokens(), s.tree_leaf_tokens()) for s in corpora[0]]
	sen_cons, sen_words = tuple(zip(*all_cons))
	cons_list, words_list = [x for s in sen_cons for x in s], [w for s in sen_words for w in s]
	cons, words = set(cons_list), set(words_list)
	vocab = cons | words

	word2ind = {w:i+1 for i,w in enumerate(vocab)}
	word2ind.update({'UNK':UNK_TOK})

	# Format corpus data for the GCN
	data_splits = []
	for corpus in corpora:
		d = [(s.adjacency_matrix(), s.negation_cue(), s.negation_scope(leaves_only=True)) for s in corpus]
		d = [(A, [word2ind.get(w, UNK_TOK) for w in ts], mask, cue, scope) for ((A, ts, mask), cue, scope) in d]
		data_splits.append(d)

	return data_splits, word2ind

def main():
	# Retrieve data
	(train, dev, test), word2ind = get_data()

	# pdb.set_trace()

	# Build model
	model = build_model(word2ind)

	# Train model
	train_model(model, train, dev)

	# Test model
	test_model(model, test)


if __name__ == '__main__':
	main()