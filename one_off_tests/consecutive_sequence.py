import numpy as np 
import random 

seed = xrange(100)
def getRow(num, length=6, i_max=100):
	seq_list = []
	for i in range(num):
		start = random.choice(seed)
		if start + length >= i_max:
			start -= length
		seq = range(start, start + length)
		seq_list += [seq]
	return seq_list

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class SingleRNN(Chain):
	def __init__(self):
		super(SingleRNN, self).__init__(
			embed=L.EmbedID(100, 16),
			l1=L.LSTM(16, 16),
			out=L.Linear(16, 100),
		)
		for param in self.params():
			param = param.data
			param[:] = np.random.uniform(-0.08, 0.08, param.shape)

	def reset_state(self):
		self.l1.reset_state()

	def forward(self, cur_char):
		x = self.embed(cur_char)
		h1 = self.l1(x)
		y = self.out(h1)
		return y

	def train(self, x):
		self.reset_state()
		x = np.array(x, dtype=np.int32)
		r = range(x.shape[1])
		loss = 0
		for x_in, x_out in zip(r[:len(r) - 1], r[1:]):
			seq_in = x[:,[x_in]]
			seq_out = x[:,[x_out]]
			seq_out = seq_out.ravel()
			var_in = chainer.Variable(seq_in)
			var_out = chainer.Variable(seq_out)
			out = self.forward(var_in)
			loss += F.softmax_cross_entropy(out, var_out)
		return loss

	def predict(self, x, num=3):
		self.reset_state()
		x = np.array(x, dtype=np.int32)
		r = range(x.shape[1])
		# encode stuff
		for x_in in r:
			seq_in = x[:,[x_in]]
			var_in = chainer.Variable(seq_in)
			out = self.forward(var_in)
		# now get the new sequence
		results = None
		for i in range(num):
			probability = F.softmax(out).data
			chosen = np.argmax(probability, axis=1)
			chosen.shape = (chosen.shape[0], 1)
			if results is not None:
				results = np.concatenate((results, chosen), axis=1)
			else:
				results = chosen
			out = self.forward(chainer.Variable(chosen.astype(np.int32)))
		return results

net = SingleRNN()
optimizer = optimizers.RMSprop(lr=0.01)
optimizer.setup(net)

# print net.train([[0, 1, 2, 3], [11, 12, 13, 14]]).data

for i in range(500):
	if i % 100 == 0: 
		print "iter: %d" % i
	seq = getRow(64)
	net.zerograds()
	loss = net.train(seq)
	print loss.data
	loss.backward()
	optimizer.update()

testArr = [[1, 2, 3], [14, 15, 16]]
print net.predict(testArr, num=15)
