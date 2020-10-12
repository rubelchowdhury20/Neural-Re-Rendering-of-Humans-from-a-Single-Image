# standard imports
import random

# third party imports
import cv2
import numpy as np
from PIL import Image

import torch
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
		source_image_name = self.image_list[index].split(".")[0]
		target_image_name = random.choice(self.image_list).split(".")[0]

		source_image_path = self.path + "model_images/" + source_image_name + ".jpg"
		source_dense_path = self.path + "model_images_dense/" + source_image_name + ".npy"
		source_texture_path = self.path + "model_images_atlas_texture/" + source_image_name + ".jpg"
		
		target_image_path = self.path + "model_images/" + target_image_name + ".jpg"
		target_dense_path = self.path + "model_images_dense/" + target_image_name + ".npy"


		source_image = Image.open(source_image_path)
		source_dense = np.load(source_dense_path)
		source_texture = Image.open(source_texture_path)

		target_image = Image.open(target_image_path)
		target_dense = np.load(target_dense_path)

		# converting the dense info to desirable resized format
		source_dense = np.moveaxis(source_dense, 0, -1)
		source_dense = cv2.resize(source_dense, dsize=(256, 256), interpolation = cv2.INTER_NEAREST)
		target_dense = np.moveaxis(target_dense, 0, -1)
		target_dense = cv2.resize(target_dense, dsize=(256, 256), interpolation = cv2.INTER_NEAREST)

		if self.transform:
			source_texture = self.transform(source_texture)
			source_dense = torch.from_numpy(source_dense)
			target_dense = torch.from_numpy(target_dense)
			return source_texture, source_dense, target_dense

		return source_texture, source_dense, target_dense

