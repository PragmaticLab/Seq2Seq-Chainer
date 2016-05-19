import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class SingleRNN(Chain):
	def __init__(self, dim):
		super(SingleRNN, self).__init__(
			embed=L.EmbedID(dim, 100),
			mid=L.LSTM(100, 50),
			out=L.Linear(50, dim),
		)
		for param in self.params():
			param = param.data
			param[:] = np.random.uniform(-0.08, 0.08, param.shape)

	def reset_state(self):
		self.mid.reset_state()

	def __call__(self, cur_char):
		x = self.embed(cur_char)
		h = self.mid(x)
		y = self.out(h)
		return y

	def train(self, x_np, y_np):
		self.reset_state()
		question = chainer.Variable(x_np[:len(x_np) - 1])
		self.__call__(question)
		answer_np = np.concatenate((x_np[len(x_np)-1: len(x_np)], y_np))
		answer_x = chainer.Variable(answer_np[:len(answer_np) - 1])
		answer_y = chainer.Variable(answer_np[1:])
		pred = self.__call__(answer_x)
		# print pred.data.shape
		# return F.softmax_cross_entropy(pred, answer_y)






