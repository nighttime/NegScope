from data import *
import sys
import os
import pdb
import re


def clean_lines(lines):
	cleaned = []
	for line in lines:
		cleaned_line = re.sub(r'n\'t', r'not', line)
		cleaned.append(cleaned_line)
	return cleaned

def strip_ccg_parse(line):
	original = line
	# in L nodes, remove the repeated category at closing
	line = re.sub(r'(<L)(.*?\s)\S*?(>)', r'\1\2\3', line)
	# in T nodes, remove all but the contents
	line = re.sub(r'<T(.*?)\d\s\d>', r'\1', line)
	# in L nodes, replace the word with *
	line = re.sub(r'(<L)(.*?)\S*?(\s*?>)', r'\1\2*\3', line)
	# in L nodes, remove all but the contents
	line = re.sub(r'<L|POS|>', r'', line)
	# replace all category parens with angle brackets (easier for parsing later)
	#   => replace all parens, then change back the brackets belonging to nodes
	line = line.replace('(', '<').replace(')', '>')
	line = re.sub(r'(^|\s)<(?=\s)', r'\1(', line)
	line = re.sub(r'(\s)>(?=\s)', r'\1)', line)
	# remove all whitespace
	line = line.replace(' ', '')
	return line

def strip_ccg(fname):
	lines = []
	with open(fname) as file:
		for line in file:
			if line.startswith('ID='):
				continue
			line = strip_ccg_parse(line).replace('\n', '')
			lines.append(line)
		return lines

def write_out(lines, fname):
	with open(fname, 'w+') as outfile:
		for line in lines:
			outfile.write(line)
			outfile.write('\n')

def main():
	args = sys.argv
	if args[1] == '--print-sents':
		corpora = get_word_data()
		folder_name = 'corpus_words'
		fnames = ['train.txt', 'dev.txt', 'test_A.txt', 'test_B.txt']
		assert len(fnames) == len(corpora)
		if not os.path.exists(folder_name):
			os.makedirs(folder_name)
		for i,fname in enumerate(fnames):
			lines = [' '.join(words) for words in corpora[i]]
			cleaned = clean_lines(lines)
			write_out(cleaned, folder_name + '/' + fname)

	elif args[1] == '--strip-ccg':
		infile, outfile = args[2:]
		lines = strip_ccg(infile)
		write_out(lines, outfile)

	else:
		print('Usage: corpus_2_ccg.py [--print-sents] [--strip-ccg]')
		exit(1)


if __name__ == '__main__':
	main()