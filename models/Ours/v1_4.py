import torch
import torch.nn as nn

from .common import *


class Layer(nn.Module):
	def __init__(self, channels, n_feats, reduction=4):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats

		act = nn.ReLU(inplace=True)

		self.layer1 = DoubleConv2DFeatsDim(channels, n_feats, act=act)
		self.layer2 = CALayer2DFeatsDimWithChansPosEmbed(channels, n_feats, reduction=4, act=act)

	
	def forward(self, x):
		out = self.layer2(self.layer1(x)) + x
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


class OursV14(nn.Module):
	def __init__(self, channels, scale_factor):
		super().__init__()
		
		n_feats = 64
		embed_chans = 3
		kernel_size=3
		reduction = 4

		n_blocks = 3
		n_layers = 4

		band_mean = band_means['CAVE']

		self.band_mean = torch.autograd.Variable(torch.FloatTensor(band_mean)).view([1, channels, 1, 1])

		# head
		self.conv_head = nn.Conv3d(1, n_feats, kernel_size=(embed_chans, kernel_size, kernel_size), padding=(embed_chans//2, kernel_size // 2, kernel_size // 2))

		# body
		self.blocks = nn.ModuleList()
		for i in range(n_blocks):
			self.blocks.append(Block(channels, n_feats, reduction=reduction, n_layers=n_layers))

		# tail
		self.conv_tail = nn.Sequential(
			nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3,2+scale_factor,2+scale_factor), stride=(1,scale_factor,scale_factor), padding=(1,1,1)),
			nn.Conv3d(n_feats, 1, kernel_size, padding=kernel_size//2)
		)

	def forward(self, x):
		out = x - self.band_mean.to(x.device)
		out = out.unsqueeze(1)

		# head
		out_head = self.conv_head(out)

		# blocks
		out = out_head
		for block in self.blocks:
			out = block(out)

		# tail
		out = self.conv_tail(out + out_head)

		out = out.squeeze(1)
		out = out + self.band_mean.to(x.device)
		return out
