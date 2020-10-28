# third party imports
import torch
import torch.nn as nn

# local imports
from . import feature_net
from . import feature_render
from . import pix2pixHD_model

class CreateModel(nn.Module):
	def __init__(self, config):
		super(CreateModel, self).__init__()
		self.config = config
		self.feature_net = feature_net.FeatureNet(num_classes=self.config.args.netG_input_nc, up_mode="upsample").to(self.config.DEVICE)
		self.feature_render = feature_render.FeatureRender(self.config).to(self.config.DEVICE)
		self.render_net = pix2pixHD_model.Pix2PixHDModel(config.args).to(self.config.DEVICE)
		if config.args.is_train and len(config.args.gpu_ids):
			self.render_net = torch.nn.DataParallel(self.render_net, device_ids=config.args.gpu_ids)
		self.optimizer_G = self.render_net.module.optimizer_G
		self.optimizer_D = self.render_net.module.optimizer_D
		# self.save = self.render_net.module

	def forward(self, batch):
		source_image = batch[0].to(self.config.DEVICE)
		source_dense = batch[1].to(self.config.DEVICE)
		source_texture = batch[2].to(self.config.DEVICE)
		target_image = batch[3].to(self.config.DEVICE)
		target_dense = batch[4].to(self.config.DEVICE)
		target_texture = batch[5].to(self.config.DEVICE)

		source_feature_output, feature_loss = self.feature_net(source_texture)
		target_feature_output, _ = self.feature_net(target_texture)
		rendered_src_feat_on_tgt = self.feature_render(source_feature_output, target_dense)
		rendered_tgt_feat_on_tgt = self.feature_render(target_feature_output, target_dense)
		rendered_src_tex_on_tgt = self.feature_render(source_texture, target_dense)

		loss_D_fake, loss_D_real, loss_G_GAN, loss_G_VGG = self.render_net(source_image, rendered_src_feat_on_tgt, target_image, rendered_tgt_feat_on_tgt, rendered_src_tex_on_tgt)

		loss_D = loss_D_fake + loss_D_real

		return feature_loss, loss_D, loss_G_GAN,  loss_G_VGG