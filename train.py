# standard imports
import os
import math
import time
import glob
import random

# third party imports
import torch
from torch.utils.data import DataLoader

import numpy as np

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

	# epoch options
	iter_path = os.path.join(config.args.checkpoints_dir, config.args.name, 'iter.txt')
	if config.args.continue_train:
		try:
			start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
		except:
			start_epoch, epoch_iter = 1, 0
		print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
	else:    
		start_epoch, epoch_iter = 1, 0



	# loading dataloader
	train_dataset = data_loader.NeuralDataset(train_image_list, data_path, config.data_transforms["train"])
	train_loader = DataLoader(train_dataset, **config.PARAMS)
	dataset_size = len(train_loader)

	# initializng the model
	model = create_model.CreateModel(config)
	if config.args.is_train and len(config.args.gpu_ids):
		model = torch.nn.DataParallel(model, device_ids=config.args.gpu_ids)

	optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

	# saving options
	total_steps = (start_epoch-1) * dataset_size + epoch_iter
	save_delta = total_steps % config.args.save_latest_freq


	for epoch in range(start_epoch, config.args.niter + config.args.niter_decay + 1):
		epoch_start_time = time.time()
		if epoch != start_epoch:
			epoch_iter = epoch_iter % dataset_size
		for idx, batch in enumerate(train_loader):

			total_steps += config.args.batch_size
			epoch_iter += config.args.batch_size
			
			feature_loss, loss_D, loss_G_GAN, loss_G_VGG = model(batch)

			 # calculate final loss scalar
			loss_G = config.args.lambda_tex * feature_loss + config.args.lambda_adv * loss_G_GAN + config.args.lambda_vgg * loss_G_VGG

			############### Backward Pass ####################
			# update generator weights
			optimizer_G.zero_grad()
			loss_G.backward(retain_graph=True)          
			optimizer_G.step()

			# update discriminator weights
			optimizer_D.zero_grad()
			loss_D.backward(retain_graph=True)        
			optimizer_D.step()


			### save latest model
			if total_steps % config.args.save_latest_freq == save_delta:
				print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
				model.module.render_net.module.save('latest')            
				np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

		### save model for this epoch
		if epoch % config.args.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
			model.module.save('latest')
			model.module.save(epoch)
			np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

		### linearly decay learning rate after certain iterations
		if epoch > config.args.niter:
			model.module.update_learning_rate()



