# standard imports
import random

# third party imports
import cv2
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# local imports
from config import texture_transform


class NeuralDataset(Dataset):
	def __init__(self, image_list, path, transform = None):
		self.image_list = image_list
		self.path = path
		self.transform = transform

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, index):
		apparel_name = self.image_list[index][2]
		if(random.choice([True, False])):
			source_image_name = self.image_list[index][0][:-4]
			target_image_name = self.image_list[index][1][:-4]
		else:
			source_image_name = self.image_list[index][1][:-4]
			target_image_name = self.image_list[index][0][:-4]

		source_image_path = self.path + "lip_images/" + source_image_name + ".jpg"
		source_dense_path = self.path + "lip_dense/" + source_image_name + ".npy"
		source_texture_path = self.path + "lip_textures/" + source_image_name + ".jpg"
		
		target_image_path = self.path + "lip_images/" + target_image_name + ".jpg"
		target_dense_path = self.path + "lip_dense/" + target_image_name + ".npy"
		target_texture_path = self.path + "lip_textures/" + target_image_name + ".jpg"

		apparel_path = self.path + "lip_apparels/" + apparel_name
		apparel_image = Image.open(apparel_path)

		source_image = Image.open(source_image_path)
		source_dense = np.load(source_dense_path)
		source_texture = Image.open(source_texture_path)

		target_image = Image.open(target_image_path)
		target_dense = np.load(target_dense_path)
		target_texture = Image.open(target_texture_path)

		# converting the dense info to desirable resized format
		source_dense = np.moveaxis(source_dense, 0, -1)
		source_dense = cv2.resize(source_dense, dsize=(256, 256), interpolation = cv2.INTER_NEAREST)
		target_dense = np.moveaxis(target_dense, 0, -1)
		target_dense = cv2.resize(target_dense, dsize=(256, 256), interpolation = cv2.INTER_NEAREST)

		if self.transform:
			source_image = self.transform(source_image)
			source_dense = torch.from_numpy(source_dense)
			source_texture = texture_transform(source_texture)

			target_image = self.transform(target_image)
			target_dense = torch.from_numpy(target_dense)
			target_texture = texture_transform(target_texture)

			apparel_image = self.transform(apparel_image)

			return source_image, source_dense, source_texture, target_image, target_dense, target_texture, apparel_image

		return source_image, source_dense, source_texture, target_image, target_dense, target_texture, apparel_image

