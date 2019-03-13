import numpy as np
import scipy.sparse as sp
import torch
import pdb

def normalize(mx):
	"""Row-normalize sparse matrix"""
	rowsum = np.array(mx.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	pdb.set_trace()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	pdb.set_trace()
	return mx


a = np.arange(16).reshape((4,4)).astype(float)

norma = normalize(a)

pdb.set_trace()