from collections import defaultdict
import pdb

from ParseTree import *




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
			# [chapter, sen_num, tok_num, word, lemma, pos, syntax, [neg_trig, scope, event]* ]

			ch, sen_num = l[0], int(l[1])

			# Read parse tree data
			chapters[ch][sen_num][0].append(l[3])
			chapters[ch][sen_num][1].append(l[5])
			chapters[ch][sen_num][2] += l[6]
			
			# Read negation data
			if l[7] != '***':
				n1 = ['' if x == '_' else x for x in l[7:10]]
				chapters[ch][sen_num][3][0].append(n1)
				if len(l) > 10:
					n2 = ['' if x == '_' else x for x in l[10:13]]
					chapters[ch][sen_num][3][1].append(n2)


	# Simplify chapter data

	# {chapter_name: [sent_vals]}
	chapter_sents = {name: [vals for _,vals in sorted(sents.items(), key=lambda x: x[0])] for name,sents in chapters.items()}

	# Construct constituent trees from input data

	# {chapter_name: [ParseTree, [neg_scope1, neg_scope2]]}
	chapter_trees = {name: [(ParseTree(*vals[:3]), vals[3]) for vals in sents] for name,sents in chapter_sents.items()}

	# pdb.set_trace()

	return chapter_trees
			