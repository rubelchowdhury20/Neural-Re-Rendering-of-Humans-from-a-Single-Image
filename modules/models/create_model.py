# third party imports
import torch
import torch.nn as nn

# local imports
from . import feature_net
from . import feature_render
from . import pix2pixHD_model
from .base_model import BaseModel

class CreateModel(BaseModel):
	def __init__(self, config):
		super(CreateModel, self).__init__()
		self.config = config

		BaseModel.initialize(self, self.config.args)

		# self.feature_net = feature_net.FeatureNet(num_classes=self.config.args.feature_output_nc, depth=self.config.args.feature_depth, up_mode="upsample").to(self.config.DEVICE)
		# self.feature_render = feature_render.FeatureRender(self.config).to(self.config.DEVICE)
		# self.render_net = pix2pixHD_model.Pix2PixHDModel(self.config.args).to(self.config.DEVICE)


		self.feature_net = feature_net.FeatureNet(num_classes=self.config.args.feature_output_nc,
													depth=self.config.args.feature_depth,
													up_mode="upsample").to("cuda:0")
		self.feature_render = feature_render.FeatureRender(self.config).to("cuda:0")
		self.render_net = pix2pixHD_model.Pix2PixHDModel(self.config.args).to("cuda:1")



		if config.args.is_train and len(config.args.gpu_ids):
			pass
			# self.feature_net = torch.nn.DataParallel(self.feature_net, device_ids=config.args.gpu_ids)
			# self.feature_render = torch.nn.DataParallel(self.feature_render, device_ids=config.args.gpu_ids)
			# self.render_net = torch.nn.DataParallel(self.render_net, device_ids=config.args.gpu_ids)

		# load networks
		if not self.config.args.is_train or self.config.args.continue_train or self.config.args.load_pretrain:
			pretrained_path = '' if not self.config.args.is_train else self.config.args.load_pretrain
			self.load_network(self.feature_net, 'Feature', self.config.args.which_epoch, pretrained_path)



		self.optimizer_feature = torch.optim.Adam(self.feature_net.parameters(), lr=self.config.args.lr, betas=(self.config.args.beta1, 0.999)) 
		self.optimizer_G = self.render_net.optimizer_G
		self.optimizer_D = self.render_net.optimizer_D

	def forward(self, batch):
		# source_image = batch[0].to(self.config.DEVICE)
		# source_dense = batch[1].to(self.config.DEVICE)
		# source_texture = batch[2].to(self.config.DEVICE)
		# target_image = batch[3].to(self.config.DEVICE)
		# target_dense = batch[4].to(self.config.DEVICE)
		# target_texture = batch[5].to(self.config.DEVICE)
		# apparel_image = batch[6].to(self.config.DEVICE)

		source_image = batch[0].to("cuda:0")
		source_dense = batch[1].to("cuda:0")
		source_texture = batch[2].to("cuda:0")
		target_image = batch[3].to("cuda:0")
		target_dense = batch[4].to("cuda:0")
		target_texture = batch[5].to("cuda:0")
		apparel_image = batch[6].to("cuda:1")

		source_background_mask = torch.logical_not(source_dense[:,:,:,0] == 0)
		target_background_mask = torch.logical_not(target_dense[:,:,:,0] == 0)
		source_background_mask = source_background_mask.unsqueeze(1).repeat(1, source_image.shape[1], 1, 1)
		target_background_mask = target_background_mask.unsqueeze(1).repeat(1, target_image.shape[1], 1, 1)
		source_image = source_image * source_background_mask
		target_image = target_image * target_background_mask


		source_feature_output, feature_loss = self.feature_net(source_texture)
		target_feature_output, _ = self.feature_net(target_texture)
		rendered_src_feat_on_tgt = self.feature_render(source_feature_output, target_feature_output, target_dense, source_texture, target_image)
		rendered_tgt_feat_on_tgt = self.feature_render(target_feature_output, target_feature_output, target_dense, target_texture, target_image)
		# rendered_src_tex_on_tgt = self.feature_render(source_texture, target_dense)

		rendered_src_feat_on_tgt = rendered_src_feat_on_tgt.to("cuda:1")
		rendered_tgt_feat_on_tgt = rendered_tgt_feat_on_tgt.to("cuda:1")
		source_image = source_image.to("cuda:1")
		target_image = target_image.to("cuda:1")
		apparel_image = apparel_image.to("cuda:1")


		loss_D_fake, loss_D_real, loss_G_GAN, loss_G_VGG, rendered_image = self.render_net(source_image, rendered_src_feat_on_tgt, target_image, rendered_tgt_feat_on_tgt, apparel_image)

		loss_D = loss_D_fake + loss_D_real

		return feature_loss, loss_D, loss_G_GAN,  loss_G_VGG, rendered_image, source_image, target_image, rendered_src_feat_on_tgt, rendered_tgt_feat_on_tgt

	def save_feature_net(self, which_epoch):
		self.save_network(self.feature_net, 'Feature', which_epoch, self.config.args.gpu_ids)

	def update_learning_rate(self):
		self.render_net.update_learning_rate()
		