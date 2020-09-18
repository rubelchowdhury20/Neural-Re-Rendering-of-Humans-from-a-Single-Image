# standard imports
import math
import glob
import random

# third party imports
from torch.utils.data import DataLoader

# local imports
import config
from modules import data_loader
from modules.models import create_model

def train(config):
	# important variables
	data_path = config.args.data_directory

	# steps for preparing and splitting the data for training
	image_list = [i.split("/")[-1] for i in glob.glob(data_path + "model_images/*")]

	# splitting the images to train and validation set
	random.shuffle(image_list)
	train_image_list = image_list[:math.floor(len(image_list) * 0.8)]
	val_image_list = image_list[math.ceil(len(image_list) * 0.8):]


	train_dataset = data_loader.NeuralDataset(train_image_list, data_path, config.data_transforms["train"])
	train_loader = DataLoader(train_dataset, **config.PARAMS)

	# initializng the model
	model = create_model.CreateModel()




	for idx, batch in enumerate(train_loader):
		texture_batch = batch
		feature_output, feature_loss = model(texture_batch)


