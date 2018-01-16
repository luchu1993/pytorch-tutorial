import glob
from os import path

def findFiles(path):
	return glob.glob(path)

print(findFiles('data/names/*.txt'))


import unicodedata
import string


all_letters = string.ascii_letters + '.,;'
n_letters = len(all_letters)


def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters
	)

print(unicodeToAscii('Ślusàrski'))


category_lines = {}
all_categories = []


def readLines(filename):
	lines = open(filename).read().strip().split('\n')

	return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/names/*.txt'):
	category = path.split(filename)[-1].split('.')[0]
	all_categories.append(category)
	lines = readLines(filename)
	category_lines[category] = lines


n_categories = len(all_categories)


# Turning names into Tensors
import torch


def letterToIndex(letter):
	return all_letters.find(letter)


def letterToTensor(letter):
	tensor = torch.zeros(1, n_letters)
	tensor[0][letterToIndex(letter)] = 1
	return tensor


def lineToTensor(line):
	tensor = torch.zeros(len(line), 1, n_letters)
	for li, letter in enumerate(line):
		tensor[li][0][letterToIndex(letter)] = 1
	return tensor

print(letterToTensor('J'))
print(lineToTensor('Jones').size())


import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()

		self.hidden_size = hidden_size

		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(input_size + hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		combined = torch.cat((input, hidden), 1)
		hidden = self.i2h(combined)
		output = self.i2o(combined)
		output = self.softmax(output)

	def initHidden(self):
		return Variable(torch.zeros(1, self.hidden_size))

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

