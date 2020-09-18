# third party imports
import torch
import torch.nn as nn

# local imports
from . import feature_net

class CreateModel(nn.Module):
	def __init__(self):
		super(CreateModel, self).__init__()
		self.feature_net = feature_net.FeatureNet(num_classes=16, up_mode="upsample")

	def forward(self, input):
		x = self.feature_net(input)
		return x