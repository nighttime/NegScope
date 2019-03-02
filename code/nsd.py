from data import *
from nsdmodel import *
import pdb

DATA_FOLDER = '../starsem-st-2012-data/cd-sco/corpus/'
TRAINING_FILE = 'training/SEM-2012-SharedTask-CD-SCO-training-09032012.txt'
DEV_FILE = 'dev/SEM-2012-SharedTask-CD-SCO-dev-09032012.txt'
TEST_FILE_A = 'test/SEM-2012-SharedTask-CD-SCO-test-cardboard.txt'
TEST_FILE_B = 'test/SEM-2012-SharedTask-CD-SCO-test-circle.txt'

EPOCHS = 10
CONSTITUENT_FEATURES = 128
HIDDEN_UNITS = 64
NUM_CLASSES = 2
DROPOUT = 0.25

def build_model():
	model = GCN(3, CONSTITUENT_FEATURES, HIDDEN_UNITS, NUM_CLASSES, DROPOUT)
	return model

def train_model():
	pass
# 	for epoch in range(EPOCHS):
    
#     	model.train()
#     	optimizer.zero_grad()
#     	output = model(features, adj)
#     	loss_train = F.nll_loss(output[idx_train], labels[idx_train])
#     	acc_train = accuracy(output[idx_train], labels[idx_train])
#     	loss_train.backward()
#     	optimizer.step()

# 	    if not args.fastmode:
# 	        # Evaluate validation set performance separately,
# 	        # deactivates dropout during validation run.
# 	        model.eval()
# 	        output = model(features, adj)

# 	    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
# 	    acc_val = accuracy(output[idx_val], labels[idx_val])
# 	    print('Epoch: {:04d}'.format(epoch+1),
# 	          'loss_train: {:.4f}'.format(loss_train.item()),
# 	          'acc_train: {:.4f}'.format(acc_train.item()),
# 	          'loss_val: {:.4f}'.format(loss_val.item()),
# 	          'acc_val: {:.4f}'.format(acc_val.item()),
# 	          'time: {:.4f}s'.format(time.time() - t))


def test_model():
	pass
#     model.eval()
#     output = model(features, adj)
#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))



def get_data():

	# Read in corpus data
	corpora = []
	for corpus_file in [TRAINING_FILE, DEV_FILE, TEST_FILE_A]:
		print('reading in:', corpus_file)
		chapters = read_starsem_scope_data(DATA_FOLDER + corpus_file)
		sents = [sent for _,chap in chapters.items() for sent in chap]
		corpora.append(sents)

	# Build vocabulary from training data
	all_cons = [(s[0].tree_node_tokens(), s[0].tree_leaf_tokens()) for s in corpora[0]]
	sen_cons, sen_words = tuple(zip(*all_cons))
	cons_list, words_list = [x for s in sen_cons for x in s], [w.lower() for s in sen_words for w in s]
	cons, words = set(cons_list), set(words_list)
	vocab = cons | words

	# Format corpus data for the GCN
	data_splits = []
	for corpus in corpora:
		d = [((s.adjacency_matrix(), s.tree_tokens()), n) for s,n in corpus]
		data_splits.append(d)

	pdb.set_trace()

	return data_splits, vocab

def main():
	# Retrieve data
	(train, dev, test), vocab = get_data()

	pdb.set_trace()

	# Build model
	model = build_model()

	# Train model
	train_model(model, train, dev)

	# Test model
	test_model(model, test)


if __name__ == '__main__':
	main()