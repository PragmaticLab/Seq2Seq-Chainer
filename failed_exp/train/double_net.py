import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class DoubleRNN(Chain):
	def __init__(self, dim, EOS):
		super(DoubleRNN, self).__init__(
			embed=L.EmbedID(dim, 16),
			l1=L.LSTM(16, 16),
			l2=L.LSTM(16, 16),
			out=L.Linear(16, dim),
		)
		for param in self.params():
			param = param.data
			param[:] = np.random.uniform(-0.08, 0.08, param.shape)
		self.EOS = EOS

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()

	def encode(self, x_np):
		for x_i in x_np:
			x_var = chainer.Variable(np.array([x_i]))
			x = self.embed(x_var)
			h1 = self.l1(x)
		return self.l1.c

	def decode_train(self, y_np):
		y_np = np.append(y_np, [self.EOS])
		loss = None
		for a_i, a_o in zip(y_np[:len(y_np) - 1], y_np[1:]):
			i_var = chainer.Variable(np.array([a_i]))
			o_var = chainer.Variable(np.array([a_o]))
			
			
			pred = self.forward(i_var)
			this_loss = F.softmax_cross_entropy(pred, o_var)
			if loss is not None:
				loss += this_loss
			else:
				loss = this_loss
		return loss

	def train(self, x_np, y_np):
		self.reset_state()


net = DoubleRNN(10, 9)
net.decode_train(np.array([1,2,3]))
