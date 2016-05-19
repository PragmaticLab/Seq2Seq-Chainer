import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
import re
import numpy as np

class ConvCorpus():
	def __init__(self):
		self.file_dir = "../data/conv.txt"
		self.dictionary_dir = "../model/conv.dict"
		self.dictionary = corpora.Dictionary.load(self.dictionary_dir)
		self.document = open(self.file_dir)
		self.dim = len(self.dictionary.token2id.keys()) + 1
		self.unknown = self.dim - 1

	def __iter__(self):
		for line in self.document:
			if line == '' or '\t' not in line:
				continue
			line = line.lower()
			parts = line.split('\t')
			question = re.findall(r"[\w']+|[.,!?;]", parts[0])
			answer = re.findall(r"[\w']+|[.,!?;]", parts[1])
			question += ["<eos>"]
			answer += ["<eos>"]
			question_tokens = [self.dictionary.token2id[token] if token in self.dictionary.token2id.keys() else self.unknown for token in question]
			answer_tokens = [self.dictionary.token2id[token] if token in self.dictionary.token2id.keys() else self.unknown for token in answer]
			yield np.array(question_tokens, dtype=np.int32), np.array(answer_tokens, dtype=np.int32)

# corp = ConvCorpus()
# myIter = corp.__iter__()
# while True:
# 	try:
# 		myIter.next()
# 	except:
# 		break

