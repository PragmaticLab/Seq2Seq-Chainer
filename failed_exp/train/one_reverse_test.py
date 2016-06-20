import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import sys
import numpy as np
from single_net import SingleRNN

"""
This doesn't work properly atm lol because the EOS is messing it up

To get it to work, change in singlenet:

			if word_out == self.EOS:
				break
from self.EOS to 0
"""

onetoten = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)

net = SingleRNN(11, 10)
optimizer = optimizers.SGD(lr=0.2)
optimizer.setup(net)

for i in range(500):
	print i 
	net.zerograds()
	loss = net.train(onetoten, onetoten[::-1])
	loss.backward()
	optimizer.update()
	print loss.data

t = net.predict(onetoten)
print t

