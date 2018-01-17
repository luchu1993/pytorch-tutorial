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
		return output, hidden


	def initHidden(self):
		return Variable(torch.zeros(1, self.hidden_size))

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


input = Variable(letterToTensor('A'))
hidden = Variable(torch.zeros(1, n_hidden))

output, hidden = rnn(input, hidden)
print(output)


input = Variable(lineToTensor('Albert'))
hidden = Variable(torch.zeros(1, n_hidden))

output, hidden = rnn(input[0], hidden)
print(output)


# Training

def categoryFromOutput(output):
	_, top_i = output.data.topk(1)
	category_i = top_i[0][0]
	return all_categories[category_i], category_i

print(categoryFromOutput(output))


import random

def randomChoice(l):
	return l[random.randint(0, len(l)-1)]

def randomTrainingExample():
	category = randomChoice(all_categories)
	line = randomChoice(category_lines[category])
	category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
	line_tensor = Variable(lineToTensor(line))
	return category, line, category_tensor, line_tensor

for i in range(10):
	category, line, category_tensor, line_tensor = randomTrainingExample()
	print('category = ', category, '/ line = ', line)


# Training the network
criterion = nn.NLLLoss()
learning_rate = 0.005

def train(category_tensor, line_tensor):
	hidden = rnn.initHidden()

	rnn.zero_grad()

	for i in range(line_tensor.size()[0]):
		output, hidden = rnn(line_tensor[i], hidden)

	loss = criterion(output, category_tensor)
	loss.backward()


	for p in rnn.parameters():
		p.data.add_(-learning_rate, p.grad.data)

	return output, loss.data[0]


import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


# Plotting the results
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
plt.show()


# Evaluating the results
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000


def evaluate(line_tensor):
	hidden = rnn.initHidden()

	for i in range(line_tensor.size()[0]):
		output, hidden = rnn(line_tensor[i], hidden)
	return output


for i in range(n_confusion):
	category, line, category_tensor, line_tensor = randomTrainingExample()
	output = evaluate(line_tensor)
	guess, guess_i = categoryFromOutput(output)
	category_i = all_categories.index(category)
	confusion[category_i][guess_i] += 1

for i in range(n_categories):
	confusion[i] /= confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Setup axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()
