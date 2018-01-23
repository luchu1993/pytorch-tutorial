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
		output = self.o2o(output_combined)
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
		tensor[li][0][all_letters.find(letter)] = 1

	return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)


def randomTrainingExample():
	category, line = randomTrainingPair()
	category_tensor = Variable(categoryTensor(category))
	input_line_tensor = Variable(inputTensor(line))
	target_line_tensor = Variable(targetTensor(line))
	return category_tensor, input_line_tensor, target_line_tensor


# Training the network
criterion = nn.NLLLoss()
learning_rate = 0.0005


def train(category_tensor, input_line_tensor, target_line_tensor):
	hidden = rnn.initHidden()

	rnn.zero_grad()

	loss = 0

	for i in range(input_line_tensor.size()[0]):
		output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
		loss += criterion(output, target_line_tensor[i])

	loss.backward()

	for p in rnn.parameters():
		p.data.add_(-learning_rate, p.grad.data)

	return output, loss.data[0] / input_line_tensor.size()[0]


import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


rnn = RNN(n_letters, 128, n_letters)

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0

start = time.time()

for iter in range(1, n_iters+1):
	output, loss = train(*randomTrainingExample())
	total_loss += loss

	if iter % print_every == 0:
		print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter/n_iters*100, loss))

	if iter % plot_every == 0:
		all_losses.append(total_loss/plot_every)
		total_loss = 0


# Plot the losses
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)


# Sampling the network

max_length = 20

# Sample from a category and starting letter
def sample(category, start_letter='A'):
    category_tensor = Variable(categoryTensor(category))
    input = Variable(inputTensor(start_letter))
    hidden = rnn.initHidden()

    output_name = start_letter

    for i in range(max_length):
        output, hidden = rnn(category_tensor, input[0], hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        if topi == n_letters - 1:
            break
        else:
            letter = all_letters[topi]
            output_name += letter
        input = Variable(inputTensor(letter))

    return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

samples('Russian', 'RUS')

samples('German', 'GER')

samples('Spanish', 'SPA')

samples('Chinese', 'CHI')