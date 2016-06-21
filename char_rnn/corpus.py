import random
import numpy as np
import cPickle as pickle
import codecs

class Shakespeare():
	def __init__(self):
		self.file_dir = "data/input.txt"
		self.dictionary_dir = "model/input.vocab"

	def makeVocab(self):
		vocab = {} # max is 64, so vocab size is 65
		words = codecs.open(self.file_dir, 'rb', 'utf-8').read()
		words = list(words)
		dataset = np.ndarray((len(words),), dtype=np.int32)
		for i, word in enumerate(words):
			if word not in vocab:
				vocab[word] = len(vocab)
			dataset[i] = vocab[word]
		self.dim = len(vocab.items())
		pickle.dump(vocab, open(self.dictionary_dir, 'wb'))
		self.vocab = vocab
		id2char = {}
		for key, val in vocab.items():
			id2char[val] = key
		self.id2char = id2char
		self.train_data = dataset

	def getTrainingData(self, num=30, length=30):
		seed = xrange(len(self.train_data)-length)
		seq_list = []
		for i in range(num):
			start = random.choice(seed)
			seq = range(start, start + length)
			words = self.train_data[seq]
			seq_list += [words]
		return seq_list

	def encode(self, seq):
		return [self.vocab[char] for char in seq]

	def decode(self, seq):
		return "".join([self.id2char[c_id] for c_id in seq])

# corpus = Shakespeare()
# corpus.makeVocab()
# # print corpus.getTrainingData()
# print corpus.encode("apple")
# print corpus.decode(corpus.encode("apple"))
