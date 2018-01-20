import glob
from os import path
import unicodedata
import string


def findFiles(path):
	return glob.glob(path)


all_letters = string.ascii_letters + '.,;'
n_letters = len(all_letters)


def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters
	)


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

print('# categories:', n_categories, all_categories)


import torch
import torch.nn as nn
from torch.autograd import Variable


# Creating the network
class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size

		self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
		self.o2o = nn.Linear(output_size + hidden_size, output_size)
		self.dropout = nn.Dropout(0.1)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, category, input, hidden):
		input_combined = torch.cat((category, input, hidden), 1)
		hidden = self.i2h(input_combined)
		output = self.i2o(input_combined)
		output_combined = torch.cat((hidden, output), 1)
		output = self.output(output_combined)
		output = self.dropout(output)
		output = self.softmax(output)
		return output, hidden

	def initHidden(self):
		return Variable(torch.zeros(1, self.hidden_size))


# Training
import random

def randomChioce(l):
	return l[random.randint(0, len(l)-1)]


def randomTrainingPair():
	category = randomChioce(all_categories)
	line = randomChioce(category_lines[category])
	return category, line


def categoryTensor(category):
	li = all_categories.index(category)
	tensor = torch.zeros(1, n_categories)
	tensor[0][li] = 1
	return tensor


def inputTensor(line):
	tensor = torch.zeros(len(line), 1,  n_letters)

	for li in range(len(line)):
		letter = line[li]
		tensor[li][0][all_letters.index(letter)] = 1

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)
	