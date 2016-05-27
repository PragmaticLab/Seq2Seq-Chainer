import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class SingleRNN(Chain):
	def __init__(self, dim, EOS, unknown):
		super(SingleRNN, self).__init__(
			embed=L.EmbedID(dim, 100),
			l1=L.LSTM(100, 100),
			l2=L.LSTM(100, 100),
			out=L.Linear(100, dim),
		)
		for param in self.params():
			param = param.data
			param[:] = np.random.uniform(-0.08, 0.08, param.shape)
		self.EOS = EOS
		self.unknown = unknown

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()

	def forward(self, cur_char):
		x = self.embed(cur_char)
		h1 = self.l1(x)
		h2 = self.l2(h1)
		y = self.out(h2)
		return y

	def train(self, x_np, y_np):
		self.reset_state()
		# first enter in the source
		for x_i in x_np[:len(x_np) - 1]:
			x_var = chainer.Variable(np.array([x_i]))
			self.forward(x_var)
		
		# now do the loss thing on the translation
		loss = None
		answer_np = np.concatenate((x_np[len(x_np)-1: len(x_np)], y_np))
		for a_i, a_o in zip(answer_np[:len(answer_np) - 1], answer_np[1:]):
			i_var = chainer.Variable(np.array([a_i]))
			o_var = chainer.Variable(np.array([a_o]))
			pred = self.forward(i_var)
			this_loss = F.softmax_cross_entropy(pred, o_var)
			if loss is not None:
				loss += this_loss
			else:
				loss = this_loss
		return loss

	# this assumes that u hv <eos> at the end
	def predict(self, x_np):
		self.reset_state()
		for x_i in x_np[:len(x_np) - 1]:
			x_var = chainer.Variable(np.array([x_i]))
			self.forward(x_var)

		word_in = chainer.Variable(np.array([x_np[len(x_np) - 1]]))
		sentence = []
		count = 0
		while True:
			word_out = self.forward(word_in).data[0].astype(np.float64)
			word_out[word_out < 0] = 0
			word_out /= np.sum(word_out)
			word_out = np.random.choice(range(len(word_out)), p=word_out)
			sentence += [word_out]
			if word_out == self.EOS:
				break
			count += 1
			if count > 100:
				break
		return np.array(sentence, dtype=np.int32)
