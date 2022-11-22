import torch
import torch.nn as nn

from .common import *


class Layer(nn.Module):
	def __init__(self, channels, n_feats, img_size, patch_size,
							dim, input_resolution, num_heads, window_size=8, 
							mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
							drop_path=0., norm_layer=nn.LayerNorm, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats

		self.dim = dim
		self.input_resolution = input_resolution

		# Embed
		self.patch_embed = PatchEmbed(
						img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
						norm_layer=None)
		
		self.patch_unembed = PatchUnEmbed(
						img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
						norm_layer=None)

		# Swin Transformr
		self.layer1 = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
																num_heads=num_heads, window_size=window_size,
																shift_size=0 if (i % 2 == 0) else window_size // 2,
																mlp_ratio=mlp_ratio,
																qkv_bias=qkv_bias, qk_scale=qk_scale,
																drop=drop, attn_drop=attn_drop,
																drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
																norm_layer=norm_layer) for i in range(2)])
	
	def forward(self, x):
		N, F,C,H,W = x.shape
		x_size = (H, W)
		out = x.transpose(1, 2).reshape(N*C, F, H, W)

		# Swin Transformr
		out = self.patch_embed(out)

		for blk in self.layer1:
			out = blk(out, x_size)
		out = self.patch_unembed(out, x_size)

		return out.reshape(N, C, F, H, W).transpose(1, 2)


class Block(nn.Module):
	def __init__(self, channels, n_feats, img_size, patch_size, 
							dim, input_resolution, num_heads, n_layers=3, act=nn.ReLU(inplace=True)):
		super().__init__()

		self.channels = channels
		self.n_feats = n_feats

		layers = []
		for i in range(n_layers):
			layers.append(Layer(channels=channels, n_feats=n_feats, img_size=img_size, patch_size=patch_size,
							dim=dim, input_resolution=input_resolution, num_heads=num_heads))

		self.layers = nn.Sequential(*layers)

		self.conv = nn.Conv3d(n_feats, n_feats, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
		
	def forward(self, x):
		out = self.layers(x)
		out = self.conv(out) + x
		return out


class OursV21(nn.Module):
	def __init__(self, channels, scale_factor):
		super().__init__()
		
		n_feats = 32  # n_feats % 6 == 0
		embed_chans = 3
		kernel_size=3

		n_blocks = 3
		n_layers = 3

		act = nn.LeakyReLU(inplace=True)

		# band mean
		band_mean = band_means['CAVE']
		self.band_mean = torch.autograd.Variable(torch.FloatTensor(band_mean)).view([1, channels, 1, 1])

		# 1. head
		self.conv_head = nn.Conv3d(1, n_feats, kernel_size=(embed_chans, kernel_size, kernel_size), padding=(embed_chans//2, kernel_size // 2, kernel_size // 2))

		# 2. body
		img_size = 64
		patch_size = 1
		embed_dim = n_feats
		self.patch_norm = True
		norm_layer=nn.LayerNorm

		self.patch_embed = PatchEmbed(
			img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
			norm_layer=norm_layer if self.patch_norm else None)
		num_patches = self.patch_embed.num_patches
		patches_resolution = self.patch_embed.patches_resolution
		self.patches_resolution = patches_resolution

		self.patch_unembed = PatchUnEmbed(
			img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
			norm_layer=norm_layer if self.patch_norm else None)

		self.blocks = nn.ModuleList()
		for i in range(n_blocks):
			self.blocks.append(Block(channels, n_feats, 
															img_size=img_size, 
															patch_size=patch_size,
															dim=embed_dim, 
															input_resolution=(patches_resolution[0], patches_resolution[1]), 
															num_heads=6,
															n_layers=n_layers, act=act))

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
