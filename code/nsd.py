from data import *
import pdb

DATA_FOLDER = '../starsem-st-2012-data/cd-sco/corpus/'
TRAINING_FILE = 'training/SEM-2012-SharedTask-CD-SCO-training-09032012.txt'



def main():
	# Get parse tree data from the corpus
	chapters = read_starsem_scope_data(DATA_FOLDER + TRAINING_FILE)
	path = chapters['baskervilles01'][1][0].longest_syntactic_path()
	pdb.set_trace()


if __name__ == '__main__':
	main()