# standard imports
import os
import math
import time
import glob
import pickle
import random

# third party imports
import torch
from torch.utils.data import DataLoader

import numpy as np
from collections import OrderedDict

# local imports
import config
from modules.data_loader import NeuralDataset
from modules.models import create_model
from modules.utils.util import AverageMeter
from modules.utils.util import tensor2im
from modules.utils.visualizer import Visualizer



def train(config):
	# important variables
	data_path = config.args.data_directory

	# steps for preparing and splitting the data for training
	with open(data_path + "lip_pairs.pkl", "rb") as fp:   #Pickling
		image_list = pickle.load(fp)
	# image_list = [i.split("/")[-1] for i in glob.glob(data_path + "model_images/*")]

	# splitting the images to train and validation set
	random.shuffle(image_list)
	train_image_list = image_list[:math.floor(len(image_list) * 0.8)]
	val_image_list = image_list[math.ceil(len(image_list) * 0.8):]

	# epoch options
	iter_path = os.path.join(config.args.checkpoints_dir, config.args.name, 'iter.txt')
	# epoch_iter => the number of iterations in the current epoch
	# start_epoch => this is the epoch from which the training will start
	if config.args.continue_train:
		try:
			start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
		except:
			start_epoch, epoch_iter = 1, 0
		print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
	else:    
		start_epoch, epoch_iter = 1, 0



	# loading dataloader
	train_dataset = NeuralDataset(train_image_list, data_path, config.data_transforms["train"])
	train_loader = DataLoader(train_dataset, **config.PARAMS)
	dataset_size = len(train_loader)

	# initializng the model
	model = create_model.CreateModel(config)
	if config.args.is_train and len(config.args.gpu_ids):
		pass
		# model = torch.nn.DataParallel(model, device_ids=config.args.gpu_ids)

	# optimizer_feature, optimizer_G, optimizer_D = model.module.optimizer_feature, model.module.optimizer_G, model.module.optimizer_D
	optimizer_feature, optimizer_G, optimizer_D = model.optimizer_feature, model.optimizer_G, model.optimizer_D


	# initializing the visualization option
	visualizer = Visualizer(config.args)

	# saving options
	total_steps = (start_epoch-1) * dataset_size + epoch_iter	# total number of iterations/steps happened till now, from the start of the training
	save_delta = total_steps % config.args.save_latest_freq		
	display_delta = total_steps % config.args.display_freq
	print_delta = total_steps % config.args.print_freq

	feature_loss_meter = AverageMeter()
	loss_D_meter = AverageMeter()
	loss_G_GAN_meter = AverageMeter()
	loss_G_VGG_meter = AverageMeter()
	loss_G_meter = AverageMeter()


	for epoch in range(start_epoch, config.args.niter + config.args.niter_decay + 1):
		epoch_start_time = time.time()
		if epoch != start_epoch:
			epoch_iter = epoch_iter % dataset_size	# reinitializing the current epoch iterations after every epoch
		for idx, batch in enumerate(train_loader):
			with torch.autograd.set_detect_anomaly(True):
				total_steps += config.args.batch_size
				epoch_iter += config.args.batch_size
				
				feature_loss, loss_D, loss_G_GAN, loss_G_VGG, rendered_image, src_img, tgt_img, src_feat_rendered_on_tgt, tgt_feat_rendered_on_tgt = model(batch)

				feature_loss = torch.mean(feature_loss)
				loss_D = torch.mean(loss_D)
				loss_G_GAN = torch.mean(loss_G_GAN)
				loss_G_VGG = torch.mean(loss_G_VGG)

				 # calculate final loss scalar
				# loss_G = config.args.lambda_tex * feature_loss + config.args.lambda_adv * loss_G_GAN + config.args.lambda_vgg * loss_G_VGG
				feature_loss = config.args.lambda_tex * feature_loss
				loss_G = config.args.lambda_adv * loss_G_GAN + config.args.lambda_vgg * loss_G_VGG

				feature_loss_meter.update(feature_loss.item(), config.args.batch_size)
				loss_D_meter.update(loss_D.item(), config.args.batch_size)
				loss_G_GAN_meter.update(loss_G_GAN.item(), config.args.batch_size)
				loss_G_VGG_meter.update(loss_G_VGG.item(), config.args.batch_size)
				loss_G_meter.update(loss_G.item(), config.args.batch_size)

				############### Backward Pass ####################
				optimizer_G.zero_grad()
				optimizer_feature.zero_grad()
				optimizer_D.zero_grad()

				# update generator weights
				loss_G.backward(retain_graph=True)          
				optimizer_G.step()

				# update feature-net weights
				feature_loss.backward(retain_graph=True)
				optimizer_feature.step()
				
				# update discriminator weights
				loss_D.backward        
				optimizer_D.step()

				### print out errors
				if total_steps % config.args.print_freq == print_delta:
					print("Train Progress--\t"
						"Train Epoch: {} [{}/{}]\t"
						"generator Loss:{:.4f} ({:.4f})\t"
						"Discriminator Loss:{:.4f} ({:.4f})\t"

						
						"feature Loss:{:.4f} ({:.4f})\t"
						"Adversarial Loss:{:.4f} ({:.4f})\t"
						"Vgg Loss:{:.4f} ({:.4f})".format(epoch, epoch_iter, dataset_size,
																	loss_G_meter.val, loss_G_meter.avg,
																	loss_D_meter.val, loss_D_meter.avg,
																	feature_loss_meter.val, feature_loss_meter.avg,
																	loss_G_GAN_meter.val, loss_G_GAN_meter.avg,
																	loss_G_VGG_meter.val, loss_G_VGG_meter.avg))
					print("\n")

				if total_steps % config.args.display_freq == display_delta:
					fake_image = rendered_image[0].cpu().detach()
					src_on_tgt = src_feat_rendered_on_tgt[0,:3,:,:].cpu().detach()
					tgt_on_tgt = tgt_feat_rendered_on_tgt[0,:3,:,:].cpu().detach()

					src_on_tgt_masked = src_feat_rendered_on_tgt[0,-3:,:,:].cpu().detach()
					tgt_on_tgt_masked = tgt_feat_rendered_on_tgt[0,-3:,:,:].cpu().detach()

					visuals = OrderedDict([("source_image", tensor2im(src_img[0])),
											("target_image", tensor2im(tgt_img[0])),
											("rendered_image", tensor2im(fake_image)),
											("src on tgt", tensor2im(src_on_tgt)),
											("tgt on tgt", tensor2im(tgt_on_tgt)),
											("src on tgt masked", tensor2im(src_on_tgt_masked)),
											("tgt on tgt masked", tensor2im(tgt_on_tgt_masked)),
											("source texture", tensor2im(batch[2][0]))])

					visualizer.display_current_results(visuals, epoch, total_steps)

				### save latest model
				# if total_steps % config.args.save_latest_freq == save_delta:
				# 	print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
				# 	# model.module.render_net.save('latest')
				# 	# model.module.save_feature_net("latest") 

				# 	model.render_net.save('latest')
				# 	model.save_feature_net("latest")            
				# 	np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')


		### save model for this epoch
		# if epoch % config.args.save_epoch_freq == 0:
		# 	print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
			
		# 	# model.module.render_net.save('latest')
		# 	# model.module.save_feature_net("latest")            

		# 	# model.module.render_net.save(epoch)
		# 	# model.module.save_feature_net(epoch)

		# 	model.render_net.save('latest')
		# 	model.save_feature_net("latest")            

		# 	model.render_net.save(epoch)
		# 	model.save_feature_net(epoch)

		# 	np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

		## linearly decay learning rate after certain iterations
		if epoch > config.args.niter:
			# model.module.update_learning_rate()
			model.update_learning_rate()



