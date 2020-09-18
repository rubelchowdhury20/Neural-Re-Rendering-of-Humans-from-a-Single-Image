# standard imports
import random

# third party imports
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

# local imports


class NeuralDataset(Dataset):
	def __init__(self, image_list, path, transform = None):
		self.image_list = image_list
		self.path = path
		self.transform = transform

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, index):
		image_name = self.image_list[index].split(".")[0]
		target_image_name = random.choice(self.image_list).split(".")[0]

		image_path = self.path + "model_images/" + image_name + ".jpg"
		dense_path = self.path + "model_images_dense/" + image_name + ".npy"
		texture_path = self.path + "model_images_texture/" + image_name + ".jpg"
		
		target_image_path = self.path + "model_images/" + target_image_name + ".jpg"
		target_dense_path = self.path + "model_images_dense/" + target_image_name + ".npy"


		image = Image.open(image_path)
		dense = np.load(dense_path)
		texture = Image.open(texture_path)

		target_image = Image.open(target_image_path)
		target_dense = np.load(dense_path)

		if self.transform:
			return self.transform(texture)

		return texture

