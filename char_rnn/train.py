import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from corpus import Shakespeare
import numpy as np
from chainer import serializers

class SingleRNN(Chain):
	def __init__(self, dim, embedding_dim):
		super(SingleRNN, self).__init__(
			embed=L.EmbedID(dim, embedding_dim),
			l1=L.LSTM(embedding_dim, embedding_dim),
			l2=L.LSTM(embedding_dim, embedding_dim),
			out=L.Linear(embedding_dim, dim),
		)
		for param in self.params():
			param = param.data
			param[:] = np.random.uniform(-0.08, 0.08, param.shape)

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()

	def forward(self, cur_char):
		x = self.embed(cur_char)
		h1 = self.l1(x)
		h2 = self.l2(h1)
		y = self.out(h2)
		return y

	def trainOneStep(self, x, y):
		x = Variable(x)
		t = Variable(y)
		y = self.forward(x)
		return F.softmax_cross_entropy(y, t)

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

net = SingleRNN(dim=corpus.dim, embedding_dim=128)
optimizer = optimizers.RMSprop(lr=0.002, alpha=0.95)
optimizer.setup(net)

n_epochs = 50
batch_size = 100
bprop_len = 50
train_data = corpus.train_data
whole_len = train_data.shape[0]
jump = whole_len / batch_size

accum_loss = Variable(np.zeros((), dtype=np.float32))
for i in xrange(jump * n_epochs):
	x_batch = np.array([train_data[(jump * j + i) % whole_len] for j in xrange(batch_size)])
	y_batch = np.array([train_data[(jump * j + i + 1) % whole_len] for j in xrange(batch_size)])
	print x_batch
	loss = net.trainOneStep(x_batch, y_batch)
	accum_loss += loss
	if (i + 1) % bprop_len == 0:
		print "i is %d / %d, loss is %f" % (i + 1, jump * n_epochs, accum_loss.data / bprop_len)
		optimizer.zero_grads()
		accum_loss.backward()
		accum_loss.unchain_backward() 
		accum_loss = Variable(np.zeros((), dtype=np.float32))
		optimizer.clip_grads(5.0)
		optimizer.update()


print corpus.decode(net.predict([corpus.encode("Before we proceed")], num=100)[0])
print corpus.decode(net.predict([corpus.encode("My lord")], num=100)[0])


# serializers.save_npz("model/shakespeare.mod", net)
