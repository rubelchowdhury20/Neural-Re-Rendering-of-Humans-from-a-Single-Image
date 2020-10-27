# standard imports
import math

# third party imports
import numpy as np

import torch
import torch.nn as nn

# local imports


class FeatureRender(nn.Module):
	def __init__(self, config):
		super(FeatureRender, self).__init__()
		self.config = config
		
	def forward(self, feature, dense_pose):
		atlas_feature = self._unfold_texture(feature)		# dimension(batch_size, channel, 24, height, width)
		mapped_feature = self._map_texture(atlas_feature, dense_pose)
		return mapped_feature


	# unfolding the atlas feature textures to 24 channels
	def _unfold_texture(self, feature):
		self.bs, self.c, self.h, self.w = feature.shape
		self.height_fraction = math.floor(self.h/4)
		self.width_fraction = math.floor(self.w/6)
		
		unfolded_feature = feature.unfold(2, self.height_fraction, self.height_fraction).unfold(3, self.width_fraction, self.width_fraction)
		unfolded_feature = unfolded_feature.reshape(self.bs, self.c, 24, self.height_fraction, self.width_fraction)

		return unfolded_feature


	# function to transfer map the atlas texture from uv space of densepose to image pixels of human image	
	def _map_texture(self, texture, dense_pose):
		_, self.dense_h, self.dense_w, self.dense_c = dense_pose.shape

		# scattering the uv values to 25 parts based on the class information present in channel 1
		dense_scatter_zeros = torch.zeros(25, self.bs, self.dense_h, self.dense_w).to(self.config.DEVICE)
		dense_scatter_U = dense_scatter_zeros.scatter_(0, dense_pose[:,:,:,0].unsqueeze(0).long(), dense_pose[:,:,:,1].unsqueeze(0).float())
		dense_scatter_V = dense_scatter_zeros.scatter_(0, dense_pose[:,:,:,0].unsqueeze(0).long(), dense_pose[:,:,:,2].unsqueeze(0).float())


		dense_U = dense_pose[:,:,:,1].unsqueeze(0)
		dense_U = dense_U.repeat(25, 1, 1, 1)

		dense_V = dense_pose[:,:,:,2].unsqueeze(0)
		dense_V = dense_V.repeat(25, 1, 1, 1)

		dense_scatter_U[dense_scatter_U != 0] = 1
		dense_scatter_V[dense_scatter_V != 0] = 1

		dense_scatter_U = dense_scatter_U * dense_U
		dense_scatter_V = dense_scatter_V * dense_V

		dense_scatter_U = dense_scatter_U.permute(1, 0, 2, 3)
		dense_scatter_V = dense_scatter_V.permute(1, 0, 2, 3)

		# changing the range of values from uv to height and width of texture 
		dense_scatter_U = dense_scatter_U * (self.height_fraction-1)/255.
		dense_scatter_V = (255-dense_scatter_V)*(self.width_fraction-1)/255.

		# repeating the texture co-ordinate values for all the channels of texture
		dense_scatter_U = dense_scatter_U.unsqueeze(2)
		dense_scatter_U = dense_scatter_U.repeat(1, 1, self.c, 1, 1)
		dense_scatter_V = dense_scatter_V.unsqueeze(2)
		dense_scatter_V = dense_scatter_V.repeat(1, 1, self.c, 1, 1)

		# considerting the last 24 body parts ignoring the first part, which is for background
		# long conversion is needed for indexing
		dense_scatter_U = dense_scatter_U[:,1:,:,:,:].long()
		dense_scatter_V = dense_scatter_V[:,1:,:,:,:].long()

		# permuting the dimension of texture to match with the mapped desepose dimension
		texture = texture.permute(1, 0, 2, 3, 4)
		
		# retrieving the texture pixel values based on the co-ordinates calculated from uv map
		expanded_dim = self.bs * 24 * self.c
		expanded_dense_spatial_dim = self.dense_h * self.dense_w
		painted_texture = texture.reshape(expanded_dim, self.height_fraction, self.width_fraction) \
							[torch.arange(expanded_dim).repeat_interleave(expanded_dense_spatial_dim), \
							dense_scatter_U.reshape(expanded_dim * expanded_dense_spatial_dim), \
							dense_scatter_V.reshape(expanded_dim * expanded_dense_spatial_dim)]

		# rearranging the dimensions and geeting the sum all the body parts, to get the full texture on the densepose
		painted_texture = painted_texture.reshape(self.bs, 24, self.c, self.dense_h, self.dense_w)
		painted_texture = painted_texture.permute(0, 2, 1, 3, 4)
		painted_texture = torch.sum(painted_texture, 2)

		return painted_texture