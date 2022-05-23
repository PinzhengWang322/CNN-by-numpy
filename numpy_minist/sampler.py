import numpy as np
def sample_function(X, y, batch_size):
	A = np.concatenate([X, y], axis = 1).astype(np.float64)
	A = A[np.random.choice(A.shape[0], batch_size, replace=False), :]
	return np.hsplit(A, np.array([A.shape[1] - 1]))

class Sampler(object):
	def __init__(self, X, y, args):
		self.X = X
		self.y = y
		self.batch_size = args.batch_size

	def next_batch(self):
		return sample_function(self.X, self.y, self.batch_size)

if __name__ == '__main__':
	# how sample_function work
	X = np.arange(12).reshape(4,3)
	y = np.arange(4).reshape(4,1)
	print(X,y)
	print(sample_function(X, y, 3))