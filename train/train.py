import sys
import chainer
import numpy as np
sys.path.insert(0, '../preprocess/')
from corpus import ConvCorpus
from net import SingleRNN


corp = ConvCorpus()
myIter = corp.__iter__()
# while True:
# 	try:
# 		t = myIter.next()
# 		print t
# 	except Exception as e:
# 		print e
# 		break
question, answer = myIter.next()

net = SingleRNN(corp.dim)
net.train(question, answer)
