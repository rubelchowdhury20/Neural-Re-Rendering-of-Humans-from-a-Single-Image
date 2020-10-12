# third party imports
import torch
import torch.nn as nn

# local imports
from . import feature_net
from . import feature_render

class CreateModel(nn.Module):
	def __init__(self):
		super(CreateModel, self).__init__()
		self.feature_net = feature_net.FeatureNet(num_classes=16, up_mode="upsample")
		self.feature_render = feature_render.FeatureRender()

	def forward(self, source_texture, source_dense, target_dense):
		feature_output, feature_loss = self.feature_net(source_texture)
		rendered_feature = self.feature_render(feature_output, target_dense)
		return feature_output, feature_loss