from collections import defaultdict
import pdb

from ParsedSentence import *


def read_starsem_scope_data(fname):
	''' Read in training data, parse sentence syntax, and construct consituent trees'''

	# Read in annotation data from corpus
	
	# {chapter_name: {sentence_num: [sent, pos, syntax, [neg_scope1, neg_scope2]}}
	chapters = defaultdict(lambda: defaultdict(lambda: [list(), list(), str(), [list(), list()]]))

	with open(fname) as f:
		for line in f:
			if len(line) < 2:
				continue

			l = line.split('\t')
			# [chapter, sen_num, tok_num, word, lemma, pos, syntax, [neg_trig, scope, event]{0,2} ]

			ch, sen_num = l[0], int(l[1])

			# Read parse tree data
			chapters[ch][sen_num][0].append(l[3])
			chapters[ch][sen_num][1].append(l[5])
			chapters[ch][sen_num][2] += l[6]
			
			# Read negation data
			if not '***' in l[-1]:
				n1 = [None if x == '_' else x for x in l[7:10]]
				chapters[ch][sen_num][3][0].append(n1)
				if len(l) > 10:
					n2 = [None if x == '_' else x for x in l[10:13]]
					chapters[ch][sen_num][3][1].append(n2)
				

	# Simplify chapter data

	# {chapter_name: [sent_vals]}
	chapter_sents = {name: [vals for _,vals in sorted(sents.items(), key=lambda x: x[0])] for name,sents in chapters.items()}

	# Construct constituent trees from input data, duplicating sentences if they have 2 negations

	# {chapter_name: [ParsedSentence]}
	chapter_trees = {}
	for name,corpus in chapter_sents.items():
		sents = []
		for vals in corpus:
			# 2 negations in this sentence
			if vals[3][1]:
				sents.append(ParsedSentence(*vals[:3], vals[3][0]))
				sents.append(ParsedSentence(*vals[:3], vals[3][1]))
			# 1 negation in this sentence
			elif vals[3][0]:
				sents.append(ParsedSentence(*vals[:3], vals[3][0]))
			# 0 negations in this sentence
			else:
				sents.append(ParsedSentence(*vals[:3]))
		chapter_trees[name] = sents

	return chapter_trees
			