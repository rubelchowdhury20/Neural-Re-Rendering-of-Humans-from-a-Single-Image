# standard imports
import os

# third party imports
import torch
import torch.nn as nn

# local imports
from . import networks
from .base_model import BaseModel

class Pix2PixHDModel(BaseModel):
	def __init__(self, cfg):
		super(Pix2PixHDModel, self).__init__()
		self.netG_output_nc = 3
		self.cfg = cfg
		self.cfg.netG_input_nc = self.cfg.feature_output_nc + 6

		BaseModel.initialize(self, cfg)

		# define networks

		# generator network
		self.netG = networks.define_G(self.cfg.netG_input_nc, self.netG_output_nc, self.cfg.ngf, self.cfg.netG, 
						self.cfg.n_downsample_global, self.cfg.n_blocks_global, gpu_ids=self.cfg.gpu_ids)

		# discriminator network
		if self.cfg.is_train:
			use_sigmoid = self.cfg.no_lsgan
			netD_input_nc = self.cfg.netG_input_nc + self.netG_output_nc
			self.netD = networks.define_D(netD_input_nc, self.cfg.ndf, self.cfg.n_layers_D, self.cfg.norm, use_sigmoid, 
										  self.cfg.num_D, not self.cfg.no_ganFeat_loss, gpu_ids=self.cfg.gpu_ids)




		# load networks
		if not self.cfg.is_train or self.cfg.continue_train or self.cfg.load_pretrain:
			pretrained_path = '' if not self.cfg.is_train else self.cfg.load_pretrain
			self.load_network(self.netG, 'G', self.cfg.which_epoch, pretrained_path)            
			if self.cfg.is_train:
				self.load_network(self.netD, 'D', self.cfg.which_epoch, pretrained_path)  
			


		if self.cfg.is_train:
			self.old_lr = self.cfg.lr


			# set loss function and optimizers
			self.criterionGAN = networks.GANLoss(use_lsgan=not self.cfg.no_lsgan)
			self.criterionVGG = networks.VGGLoss(self.cfg.gpu_ids)

			self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, 0.999))                            
			self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, 0.999))

	def discriminate(self, input_label, test_image):
		input_concat = torch.cat((input_label, test_image.detach()), dim=1)
		return self.netD.forward(input_concat)




	def forward(self, src_img, src_rendered_on_tgt, tgt_img, tgt_rendered_on_tgt, apparel_image):
		# fake generation
		fake_image = self.netG.forward(torch.cat((src_rendered_on_tgt, apparel_image), dim=1))
		# fake_image = self.netG.forward(torch.cat((src_rendered_on_tgt, tgt_img), dim=1))

		# fake detection and loss
		pred_fake = self.discriminate(torch.cat((src_rendered_on_tgt, apparel_image), dim=1), fake_image)
		loss_D_fake = self.criterionGAN(pred_fake, False)

		# real detection and loss
		pred_real = self.discriminate(torch.cat((tgt_rendered_on_tgt, apparel_image), dim=1), tgt_img)
		loss_D_real = self.criterionGAN(pred_real, True)

		# GAN loss (Fake Passability Loss)
		pred_fake = self.discriminate(torch.cat((src_rendered_on_tgt, apparel_image), dim=1), fake_image)        
		# pred_fake = self.netD.forward(torch.cat((src_rendered_on_tgt, fake_image), dim=1))        
		loss_G_GAN = self.criterionGAN(pred_fake, True)

		# VGG feature matching loss
		loss_G_VGG = 0
		if not self.cfg.no_vgg_loss:
			loss_G_VGG = self.criterionVGG(fake_image, tgt_img)

		return loss_D_fake, loss_D_real, loss_G_GAN, loss_G_VGG, fake_image

	def save(self, which_epoch):
		self.save_network(self.netG, 'G', which_epoch, self.cfg.gpu_ids)
		self.save_network(self.netD, 'D', which_epoch, self.cfg.gpu_ids)

	def update_learning_rate(self):
		lrd = self.cfg.lr / self.cfg.niter_decay
		lr = self.old_lr - lrd        
		for param_group in self.optimizer_D.param_groups:
			param_group['lr'] = lr
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr
		self.old_lr = lr




