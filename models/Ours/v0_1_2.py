import math
import torch
import torch.nn as nn

from .common import *


class Layer(nn.Module):
	def __init__(self, channels, n_feats, reduction=4):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats

		act = nn.ReLU(inplace=True)

		# block1
		self.layer1 = DoubleConv2DFeatsDim(channels, n_feats, act=act)
		# self.layer2 = CALayer3DFeatsDimWithChannelEmbed(channels, n_feats, reduction=4, act=act)

		# block2
		self.layer3 = DoubleConv2DChannelDim(channels, n_feats, act=act)
		# self.layer4 = CALayer3DChannelDim(channels, n_feats, reduction=4, act=act)
	
	def forward(self, x):
		# block1
		out = self.layer1(x) + x
		# T = self.layer2(out) + x

		# block2
		out = self.layer3(out) + out
		# out = self.layer4(out) + T

		return out


class Block(nn.Module):
	def __init__(self, channels, n_feats, n_layers=3, reduction=4):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats

		act = nn.ReLU(inplace=True)

		layers = []
		for i in range(n_layers):
			layers.append(Layer(channels=channels, n_feats=n_feats, reduction=reduction))

		self.layers = nn.Sequential(*layers)

		self.conv = nn.Conv3d(n_feats, n_feats, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
		
	def forward(self, x):
		out = self.layers(x)
		out = self.conv(out) + x
		return out


class OursV012(nn.Module):
	def __init__(self, channels, scale_factor):
		super().__init__()
		
		
		kernel_size=3
		reduction = 4

		n_branch = 8
		n_feats = 64
		n_blocks = 6
		n_layers = 6

		embed_chans = math.ceil(channels / n_branch)	#  4
		chan_padding = (n_branch * embed_chans - channels) 
		self.chan_padding =  chan_padding
		
		band_mean = band_means['CAVE']
		self.band_mean = torch.autograd.Variable(torch.FloatTensor(band_mean)).view([1, channels, 1, 1])

		# head
		self.conv_head = nn.Conv3d(1, n_feats, 
			kernel_size=(embed_chans, kernel_size, kernel_size),
			stride=(embed_chans, 1, 1),
			padding=(chan_padding, kernel_size // 2, kernel_size // 2))

		# body
		self.blocks = nn.ModuleList()
		for i in range(n_blocks):
			self.blocks.append(Block(n_branch, n_feats, reduction=reduction, n_layers=n_layers))

		# up
		self.up = Upsampler(default_conv, scale_factor, n_feats)

		# tail
		self.conv_tail = nn.Conv2d(n_feats, embed_chans, kernel_size, padding=kernel_size//2)
			
		# tail
		# self.conv_tail = nn.Sequential(
		# 	nn.Conv3d(n_feats, 1, kernel_size, padding=kernel_size//2)
		# )

	def forward(self, x):
		out = x - self.band_mean.to(x.device)
		out = out.unsqueeze(1)

		# head
		out_head = self.conv_head(out)

		# blocks
		out = out_head
		for block in self.blocks:
			out = block(out)
		out = out + out_head

		# up
		N, F, G, H, W = out.shape
		out = out.transpose(1, 2).reshape(N*G, F, H, W)
		out = self.up(out)	# N*G, F, H, W

		# tail
		out = self.conv_tail(out)	# N*G, B, H, W
		_, B, H, W = out.shape
		out = out.reshape(N, G, B, H, W).reshape(N, G*B, H, W)[:, self.chan_padding:, :, :]	# N, C, H, W


		out = out + self.band_mean.to(x.device)
		return out
