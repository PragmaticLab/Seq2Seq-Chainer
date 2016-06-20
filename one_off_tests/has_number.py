import numpy as np 
import random 

seed = xrange(100)
basis = xrange(10)
def getRow(num):
	seq_list = []
	y_list = []
	for i in range(num):
		seq = random.sample(seed, 10)
		y = 0
		if set(basis).intersection(set(seq)):
			y = 1
		y_list += [y]
		seq_list += [seq]
	return y_list, seq_list

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class SingleRNN(Chain):
	def __init__(self):
		super(SingleRNN, self).__init__(
			embed=L.EmbedID(100, 2),
			l1=L.LSTM(2, 2),
			out=L.Linear(2, 2),
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

	def trainOne(self, x, y):
		self.reset_state()
		x = np.array(x, dtype=np.int32)
		y = np.array(y, dtype=np.int32)
		for col in range(x.shape[1]):
			x_col = x[:,[col]]
			x_var = chainer.Variable(x_col)
			out = self.forward(x_var)
		# now get the loss
		loss = F.softmax_cross_entropy(out, chainer.Variable(y))
		return loss

	def predict(self, x):
		self.reset_state()
		x = np.array(x, dtype=np.int32)
		for col in range(x.shape[1]):
			x_col = x[:,[col]]
			x_var = chainer.Variable(x_col)
			out = self.forward(x_var)
		return out

net = SingleRNN()
optimizer = optimizers.RMSprop(lr=0.01)
optimizer.setup(net)

# print net.trainOne([[11, 12, 13], [21, 22, 23]], [0, 0]).data

for i in range(500):
	if i % 100 == 0: 
		print "iter: %d" % i
	y, seq = getRow(20)
	net.zerograds()
	loss = net.trainOne(seq, y)
	loss.backward()
	optimizer.update()

testArr = [[12, 13, 41, 21, 44, 23, 55, 65, 99, 56], [8, 22, 41, 31, 44, 23, 55, 65, 99, 56]]
print net.predict(testArr).data
