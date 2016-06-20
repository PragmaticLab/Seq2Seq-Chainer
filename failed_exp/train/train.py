import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import sys
import numpy as np
sys.path.insert(0, '../preprocess/')
from corpus import ConvCorpus
from single_net import SingleRNN

corp = ConvCorpus()
net = SingleRNN(corp.dim, corp.EOS)
optimizer = optimizers.SGD()
optimizer.setup(net)

# print corp.getSentence(np.array([279,333,82,479,400,273], dtype=np.int32))
# t = net.predict(np.array([279,333,82,479,400,273], dtype=np.int32))
# print corp.getSentence(t)

count = 0
for i in range(100):
	corp = ConvCorpus()
	myIter = corp.__iter__()
	for question, answer in myIter:
		net.zerograds()
		# loss = net.train(question, answer)
		loss = net.train(question, question)
		loss.backward()
		optimizer.update()
		if count % 300 == 0:
			print "\n\nepoch: %d" % count
			print "loss : %f" % loss.data
			print corp.getSentence(question)
			print corp.getSentence(net.predict(question))
		count += 1
		