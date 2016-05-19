import sys
sys.path.insert(0, '../preprocess/')
from corpus import ConvCorpus








corp = ConvCorpus()
myIter = corp.__iter__()
# while True:
# 	try:
# 		t = myIter.next()
# 		print t
# 	except Exception as e:
# 		print e
# 		break
m = myIter.next()
