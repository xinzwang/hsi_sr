import torch
import torch.nn as nn

class BasicConv3d(nn.Module):
		def __init__(self, wn, in_channel, out_channel, kernel_size, stride, padding=(0,0,0)):
				super(BasicConv3d, self).__init__()
				self.conv = wn(nn.Conv3d(in_channel, out_channel,
															kernel_size=kernel_size, stride=stride,
															padding=padding))
				# self.relu = nn.ReLU(inplace=True)

		def forward(self, x):
				x = self.conv(x)
				# x = self.relu(x)
				return x


class Block3D(nn.Module):
	def __init__(self, wn, n_feats):
		super(Block3D, self).__init__()
		self.conv = nn.Sequential(
				BasicConv3d(wn, n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
				nn.ReLU(inplace=True),
				BasicConv3d(wn, n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))
		)            
			 
	def forward(self, x): 
		return self.conv(x)


class ResBlock3D(nn.Module):
	def __init__(self, channels, n_feats, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats

		self.conv = nn.Sequential(
			nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1,1)),
			act,
			nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1,1)),
		)
	
	def forward(self, x):
		# x: [N, F, C, H, W]
		N,F,C,H,W = x.shape
		out = x.transpose(1,2).reshape(N*C, F, H, W) # [N*C, F, H, W]

		return out.reshape(N,C,F,H,W).transpose(1,2)

class CALayer3DFeatsDim(nn.Module):
	def __init__(self, channels, n_feats, reduction=4, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats

		self.pool = nn.AdaptiveAvgPool2d((1, 1))

		self.fc = nn.Sequential(
			nn.Linear(n_feats * channels, n_feats * channels // reduction),
			act,
			nn.Linear(n_feats * channels // reduction, n_feats * channels),
			nn.Sigmoid(),
		)

	def forward(self, x):
		# x: [N, F, C, H, W]
		out = x.transpose(1,2) # [N, C, F, H, W]
		attn  = self.pool(out).reshape(-1, self.channels * self.n_feats)
		attn = self.fc(attn).reshape(N, self.channels, self.n_feats).unsqueeze(-1).unsqueeze(-1)
		out = out * attn
		out = out.transpose(1, 2)	# [N, F, C, H, W]
		return out


class CALayer3DChannelDim(nn.Module):
	def __init__(self, channels, n_feats, reduction=4, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats

		self.pool = nn.AdaptiveAvgPool2d((1, 1))

		self.fc = nn.Sequential(
			nn.Linear(n_feats * channels, n_feats * channels // reduction),
			act,
			nn.Linear(n_feats * channels // reduction, n_feats * channels),
			nn.Sigmoid(),
		)

	def forward(self, x):
		# x: [N, F, C, H, W]
		attn  = self.pool(out).reshape(-1, self.channels * self.n_feats)	# [N, F*C]
		attn = self.fc(attn).reshape(N, self.channels, self.n_feats).unsqueeze(-1).unsqueeze(-1)
		out = out * attn
		return out


class DoubleConv2DFeatsDim(nn.Module):
	"""???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????"""
	def __init__(self, channels, n_feats, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats
		
		self.conv = nn.Sequential(
			nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1,1)),
			act,
			nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1,1)),
		)

	def forward(self, x):
		# x: [N, F, C, H, W]
		N, F,C,H,W = x.shape
		out = x.transpose(1, 2).reshape(N*C, F, H, W)
		out = self.conv(out)
		return out.reshape(N, C, F, H, W).transpose(1, 2)


class DoubleConv2DChannelDim(nn.Module):
	"""????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????"""
	def __init__(self, channels, n_feats, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats
		
		self.conv = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=1, padding=(1,1)),
			act,
			nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=1, padding=(1,1)),
		)

	def forward(self, x):
		# x: [N, F, C, H, W]
		N, F,C,H,W = x.shape
		out = x.reshape(N*F, C, H, W)
		out = self.conv(out)
		return out.reshape(N, F, C, H, W)


class DoubleConv3DChannelDim(nn.Module):
	"""?????????????????????????????????????????????????????????????????????????????? ??????????????????????????????????????????????????????????????????????????????"""
	def __init__(self, channels, n_feats, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats
		
		self.conv = nn.Sequential(
			nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1,0,0)),
			act,
			nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1,0,0)),
		)

	def forward(self, x):
		# x: [N, F, C, H, W]
		N, F,C,H,W = x.shape
		out = self.conv(out)
		return out

class DoubleConv3dFeatsDim(nn.Module):
	"""????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????"""
	pass

class Block(nn.Module):
	def __init__(self, wn, channels, n_feats, reduction=4):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats

		self.res_block = nn.Sequential(
			wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1,1))),
			nn.ReLU(inplace=True),
			wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1,1))),
		)

		self.ca_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.ca_fc = nn.Sequential(
			nn.Linear(n_feats * channels, n_feats * channels // reduction),
			nn.Linear(n_feats * channels // reduction, n_feats * channels),
			nn.Sigmoid(),
		)

		self.tri_block = Block3D(wn, n_feats=n_feats)
		
		
	def forward(self, x):
		# N, C, F, H, W
		N, C, F, H, W = x.shape
		# 2d conv
		x0 = x.reshape(N*C, F, H, W)	# [N*C, F, H, W]
		out = self.res_block(x0) + x0
		# channel attn
		out = out.reshape(N, C, F, H, W)
		attn = self.ca_pool(out)
		attn = attn.reshape(N, C*F)
		attn = self.ca_fc(attn).reshape(N, C, F).unsqueeze(-1).unsqueeze(-1)	# [N, C, F, 1, 1]
		out = out * attn # [N, C, F, H, W]
		# 3d conv
		out = out.permute(0, 2, 1, 3, 4) # [N, F, C, H, W]
		out = self.tri_block(out)
		out = out.permute(0, 2, 1, 3, 4)	# [N, C, F, H, W]

		return out + x



class MCNetV2(nn.Module):
	def __init__(self, channels, scale_factor):
		super().__init__()
		
		n_feats = 32
		embed_chans = 3
		kernel_size=3
		reduction = 4

		n_blocks = 12	# 6 is too small

		band_mean = band_means['CAVE']
		self.band_mean = torch.autograd.Variable(torch.FloatTensor(band_mean)).view([1, channels, 1, 1])

		wn = lambda x: torch.nn.utils.weight_norm(x)

		self.conv_head = wn(nn.Conv3d(1, n_feats, kernel_size=(embed_chans, kernel_size, kernel_size), padding=(embed_chans//2, kernel_size // 2, kernel_size // 2)))

		self.blocks = nn.ModuleList()
		for i in range(n_blocks):
			self.blocks.append(Block(wn, channels, n_feats, reduction=reduction))

		self.conv_tail = nn.Sequential(
			wn(nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3,2+scale_factor,2+scale_factor), stride=(1,scale_factor,scale_factor), padding=(1,1,1))),
			wn(nn.Conv3d(n_feats, 1, kernel_size, padding=kernel_size//2))
		)

	def forward(self, x):
		out = x - self.band_mean.to(x.device)

		# head
		out = out.unsqueeze(1)
		out_head = self.conv_head(out).permute(0, 2, 1, 3, 4)	# [N, F, C, H, W] -> [N, C, F, H, W]
		
		# blocks
		out = out_head
		for block in self.blocks:
			out = block(out) + out_head

		out = out.permute(0, 2, 1, 3, 4)	# [N, C, F, H, W] -> [N, F, C, H, W]

		# tail
		out = self.conv_tail(out)
		out = out.squeeze(1)

		out = out + self.band_mean.to(x.device)
		return out





band_means = {
	'CAVE':(0.0939, 0.0950, 0.0869, 0.0839, 0.0850, 0.0809, 0.0769, 0.0762, 0.0788, 0.0790, 0.0834, 
					0.0894, 0.0944, 0.0956, 0.0939, 0.1187, 0.0903, 0.0928, 0.0985, 0.1046, 0.1121, 0.1194, 
					0.1240, 0.1256, 0.1259, 0.1272, 0.1291, 0.1300, 0.1352, 0.1428, 0.1541), #CAVE
}
