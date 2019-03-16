import numpy as np
import pdb
from utils import *

class ParsedSentence:
	class ParseNode:
		def __init__(self, constituent=None, leaf_ref=None):
			self.constituent = constituent
			self.leaf_ref = leaf_ref
			self.children = []
			self.negation_cue = False
			self.negation_scope = False

		def add_constituent(self, constituent):
			self.constituent = constituent

		def add_leaf_ref(self, leaf_ref):
			self.leaf_ref = leaf_ref

		def add_child(self, child):
			self.children.append(child)

		def is_leaf(self):
			return self.leaf_ref is not None

		def __str__(self):
			if self.is_leaf():
				return '*'
			else:
			# val = self.constituent or (self.words[int(self.leaf_ref)] + '/' + self.pos[int(self.leaf_ref)])
				return '(' + self.constituent + ''.join(str(c) for c in self.children) + ')'

		def pprint(self, words):
			begin = ''
			end = ''
			if self.is_leaf():
				if self.negation_cue:
					begin = Color.BOLD + begin
					end = end + Color.ENDC
				elif self.negation_scope:
					begin = Color.UNDERLINE + begin
					end = end + Color.ENDC
				return begin + words[self.leaf_ref] + end
			else:
				return ' '.join(c.pprint(words) for c in self.children)


	def __init__(self, words, pos, syntax, negation=None):
		self.words = [w.lower() for w in words]
		self.pos = pos
		self.syntax = ParsedSentence._identify_leaves(syntax)
		self.negation = negation

		self.root = ParsedSentence._parse(self.syntax)
		self.constituents = []
		ParsedSentence._collect_constituents(self.root, self.constituents)
		if self.negation:
			ParsedSentence._annotate_negation(self.constituents, self.negation)

	def __str__(self):
		return str(self.root)

	def pprint(self):
		print(self.root.pprint(self.words))

	def longest_syntactic_path(self):
		# Find shortest path from each constituent to each leaf node
		const_shortest_paths = np.full([len(self.constituents), len(self.words)], 0.)
		for i in range(len(self.constituents)):
			for j in range(len(self.words)):
				const_shortest_paths[i,j] = ParsedSentence._find_const_shortest_path(self.constituents[i], j)
		
		# Find shortest path from each leaf node to each other leaf node
		shortest_paths = np.full([len(self.words), len(self.words)], -1)
		for i in range(len(self.words)):
			for j in range(len(self.words)):
				best_path = np.min(const_shortest_paths[:,i] + const_shortest_paths[:,j])
				shortest_paths[i,j] = -1 if best_path == float('inf') else best_path

		return np.max(shortest_paths)

	def tree_tokens(self):
		'''List all parse tree node and leaf tokens'''
		return [self.words[c.leaf_ref] if c.is_leaf() else c.constituent for c in self.constituents]

	def tree_node_tokens(self):
		'''List all non-terminal node tokens'''
		return [c.constituent for c in self.constituents if not c.is_leaf()]

	def tree_leaf_tokens(self):
		'''List all terminal node tokens'''
		return self.words

	def adjacency_matrix(self, self_loops=True, row_normalize=True):
		'''Compute adjacency matrix of the parse tree (undirected graph). 
		A connection is added between all graph neighbors and optionally, self-loops.
		ex. (A (B C)) -> [[1, 1, 1],[1, 1, 0],[1, 0, 1]]

		Returns:
		the adjacency matrix A 
		the key used to map entries of A to constituents
		a 0/1 mask indicating which key constituents are leaves
		'''
		cons = len(self.constituents)
		A = np.zeros([cons, cons])
		con2ind = {c:i for i,c in enumerate(self.constituents)}

		for c in self.constituents:
			if self_loops:
				A[con2ind[c], con2ind[c]] = 1
			for adj in c.children:
				A[con2ind[c], con2ind[adj]] = 1
				A[con2ind[adj], con2ind[c]] = 1

		if row_normalize:
			row_sum = A.sum(axis=1)
			row_sum[row_sum==0] += 0.1 # prevent divide-by-0 errors for 0-sum rows
			A /= row_sum.reshape([len(row_sum),1])

		surface_mask = [1 if c.is_leaf() else 0 for c in self.constituents]

		return A, self.tree_tokens(), surface_mask

	def negation_cue(self):
		'''Returns a {0,1} mask over all tree tokens -- 1=cue, 0=otherwise'''
		return [1 if c.negation_cue else 0 for c in self.constituents]

	def negation_surface_scope(self):
		'''Returns a {0,1} mask over JUST tree leaves (words) -- 1=in-scope, 0=otherwise'''
		return [1 if c.negation_scope else 0 for c in self.constituents if c.is_leaf()]

	@classmethod
	def _annotate_negation(cls, constituents, negation):
		cue = {i for i,x in enumerate(negation) if x[0] is not None}
		scope = {i for i,x in enumerate(negation) if x[1] is not None}

		for c in constituents:
			if c.is_leaf():
				if c.leaf_ref in cue:
					c.negation_cue = True
				if c.leaf_ref in scope:
					c.negation_scope = True


	@classmethod
	def _find_const_shortest_path(cls, constituent, target):
		''' Finds the length of the shortest path between a constituent and a target leaf node
		returns length of path if the leaf is a descendent of the constituent, otherwise float('inf')
		'''
		if constituent.is_leaf():
			return 0 if constituent.leaf_ref == target else float('inf')
		else:
			return 1 + min(ParsedSentence._find_const_shortest_path(c, target) for c in constituent.children)



	@classmethod
	def _identify_leaves(cls, syntax):
		''' Read in serialized s-exp containing * in place of leaves
			Replace * with identifiers, i.e. 0, 1, 2
			ex. (S(NP*)(VP*)) -> (S(NP0)(VP1))
		'''
		i = 0
		ct = 0
		syntax_vals = list(syntax)
		while i < len(syntax_vals):
			if syntax_vals[i] == '*':
				syntax_vals[i] = str(ct) + ' '
				ct += 1
			i += 1
		return ''.join(syntax_vals)

	@classmethod
	def _parse(cls, syntax):
		tokens = ParsedSentence._tokenize(syntax)
		if tokens[0] != '(':
			raise Exception('Parse error: malformed syntax tree: {}'.format(syntax))
		root = ParsedSentence.ParseNode()
		ParsedSentence._parse_tree(root, tokens[1:])
		return root

	@classmethod
	def _parse_tree(cls, node, tokens):
		t = tokens.pop(0)
		if t == '(':
			new_node = ParsedSentence.ParseNode()
			ParsedSentence._parse_tree(new_node, tokens)
			node.add_child(new_node)
			ParsedSentence._parse_tree(node, tokens)
		elif t == ')':
			return
		elif t.isalpha():
			node.add_constituent(t)
			ParsedSentence._parse_tree(node, tokens)
		elif t.isnumeric():
			new_node = ParsedSentence.ParseNode(leaf_ref=int(t))
			node.add_child(new_node)
			ParsedSentence._parse_tree(node, tokens)
		else:
			raise Exception('Parse error: malformed syntax tree: {}')

	@classmethod
	def _tokenize(cls, syntax):
		tokens = []
		i = 0
		while i < len(syntax):
			c = syntax[i]

			# keep open or close marker
			if c in '()':
				tokens.append(c)
				i += 1

			# skip whitespace, etc
			elif c in ' \n':
				i += 1

			# scan ahead to capture an alphabetic symbol
			elif c.isalpha():
				curr_tok = ''
				while syntax[i].isalpha():
					curr_tok += syntax[i]
					i += 1
				tokens.append(curr_tok)

			# scan ahead to capture a numeric symbol
			elif c.isnumeric():
				curr_tok = ''
				while syntax[i].isnumeric():
					curr_tok += syntax[i]
					i += 1
				tokens.append(curr_tok)

			# otherwise, error
			else:
				raise Exception('Parse tokenization error: `{}` (char {}) in {}'.format(c, i, syntax))

		return tokens

	@classmethod
	def _collect_constituents(cls, node, constituents_list):
		constituents_list.append(node)
		for child in node.children:
			ParsedSentence._collect_constituents(child, constituents_list)


