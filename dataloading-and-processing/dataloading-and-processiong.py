import os
import torch
import pandas as pd 
from skimage import io, transform
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def show_landmarks(image, landmarks):
	plt.imshow(image)
	plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')


def show_landmarks_batch(sample_batched):
	images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
	batch_size = len(images_batch)
	im_size = images_batch.size(2)

	grid = utils.make_grid(images_batch)
	plt.imshow(grid.numpy().transpose(1,2,0))

	for i in range(batch_size):
		plt.scatter(
			landmarks_batch[i, :, 0].numpy() + i * im_size,
			landmarks_batch[i, :, 1].numpy(),
			s=10, marker='.', c='r')
		plt.title('Batch from dataloader')


class FaceLandmarksDataset(Dataset):
	def __init__(self, csv_file, root_dir, transform=None):
		self.landmarks_frame = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.landmarks_frame)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0])
		image = io.imread(img_name)
		landmarks = self.landmarks_frame.ix[idx, 1:].as_matrix().astype('float')
		landmarks = landmarks.reshape(-1, 2)
		sample = { 'image': image, 'landmarks': landmarks }

		if self.transform:
			sample = self.transform(sample)
		
		return sample


class Rescale:
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, landmarks = sample['image'], sample['landmarks']

		h, w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h/w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		img = transform.resize(image, (new_h, new_w))
		landmarks = landmarks * [new_w/w, new_h/h]

		return {'image': img, 'landmarks': landmarks}


class RandomCrop:
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):
		image, landmarks = sample['image'], sample['landmarks']

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h-new_h)
		left = np.random.randint(0, w-new_w)

		image = image[top:top+new_h, left: left+new_w]
		landmarks = landmarks - [left, top]
		return { 'image': image, 'landmarks': landmarks }


class ToTensor:
	def __call__(self, sample):
		image, landmarks = sample['image'], sample['landmarks']
		image = image.transpose((2, 0, 1))
		return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks) }


def main():
	landmarks_frame = pd.read_csv('./faces/face_landmarks.csv')
	n = 62
	img_name = landmarks_frame.ix[n, 0]
	landmarks = landmarks_frame.ix[n, 1:].as_matrix().astype('float')
	landmarks = landmarks.reshape(-1, 2)

	print('Image name: {}'.format(img_name))
	print('Landmarks shape: {}'.format(landmarks.shape))
	print('First 4 landmarks: {}'.format(landmarks[:4]))

	plt.figure()
	show_landmarks(io.imread(os.path.join('./faces/', img_name)), landmarks)
	plt.show()


	face_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv', root_dir='faces')
	fig = plt.figure()

	for i in range(len(face_dataset)):
		sample = face_dataset[i]
		print(i, sample['image'].shape, sample['landmarks'].shape)

		ax = plt.subplot(1, 4, i+1)
		plt.tight_layout()
		ax.set_title('Sample #{}'.format(i))
		ax.axis('off')
		show_landmarks(**sample)

		if i == 3:
			plt.show()
			break


	# compose transforms
	scale = Rescale(256)
	crop = RandomCrop(128)
	composed = transforms.Compose([Rescale(256), RandomCrop(224)])

	fig = plt.figure()
	sample = face_dataset[8]
	print(sample['image'].shape)
	for i, tsfrm in enumerate([scale, crop, composed]):
		transformed_sample = tsfrm(sample)

		ax = plt.subplot(1, 3, i+1)
		plt.tight_layout()
		ax.set_title(type(tsfrm).__name__)
		show_landmarks(**transformed_sample)

	plt.show()


	# Iterating through the dataset
	transformed_dataset = FaceLandmarksDataset(
		csv_file='faces/face_landmarks.csv',
		root_dir='faces/',
		transform=transforms.Compose([
			Rescale(256),
			RandomCrop(224),
			ToTensor()
		]))

	dataloader = DataLoader(transformed_dataset, batch_size=8, shuffle=True, num_workers=8)
	for i_batch, sample_batched in enumerate(dataloader):
		print(i_batch, sample_batched['image'].size(), sample_batched['landmarks'].size())
		if i_batch == 5:
			plt.figure()
			show_landmarks_batch(sample_batched)
			plt.axis('off')
			plt.ioff()
			plt.show()
			break


if __name__ == '__main__':
	main()