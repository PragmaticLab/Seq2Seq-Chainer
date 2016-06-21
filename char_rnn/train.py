import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from corpus import Shakespeare
import numpy as np

class SingleRNN(Chain):
	def __init__(self, dim, embedding_dim):
		super(SingleRNN, self).__init__(
			embed=L.EmbedID(dim, embedding_dim),
			l1=L.LSTM(embedding_dim, embedding_dim),
			out=L.Linear(embedding_dim, dim),
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

corpus = Shakespeare()
corpus.makeVocab()

net = SingleRNN(dim=corpus.dim, embedding_dim=64)
optimizer = optimizers.RMSprop(lr=0.01)
optimizer.setup(net)

for i in range(1000):
	if i % 100 == 0: 
		print "iter: %d" % i
	seq = corpus.getTrainingData(num=128, length=32)
	net.zerograds()
	loss = net.train(seq)
	print loss.data
	loss.backward()
	optimizer.update()

print corpus.decode(net.predict([corpus.encode("Before we proceed")], num=100)[0])
print corpus.decode(net.predict([corpus.encode("My lord")], num=100)[0])


