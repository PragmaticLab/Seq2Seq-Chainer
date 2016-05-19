import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
import re

questions = []
answers = []

f = open('../data/conv.txt', 'r')
for line in f:
	if line == '' or '\t' not in line:
		continue
	line = line.lower()
	parts = line.split('\t')
	question = re.findall(r"[\w']+|[.,!?;]", parts[0])
	answer = re.findall(r"[\w']+|[.,!?;]", parts[1])
	question += ["<eos>"]
	answer += ["<eos>"]
	questions += [question]
	answers += [answer]

assert len(questions) == len(answers)

dictionary = corpora.Dictionary(questions)
dictionary.add_documents(answers)

dictionary.filter_extremes(no_below=10, no_above=1)
dictionary.save('../model/conv.dict')
print dictionary.token2id["<eos>"]
