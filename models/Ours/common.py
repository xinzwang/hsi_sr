"""
TODO: 给feat维度的channel-attention增加光谱维度embed编码
"""
import math
import torch
import torch.nn as nn


band_means = {
	'CAVE':(0.0939, 0.0950, 0.0869, 0.0839, 0.0850, 0.0809, 0.0769, 0.0762, 0.0788, 0.0790, 0.0834, 
					0.0894, 0.0944, 0.0956, 0.0939, 0.1187, 0.0903, 0.0928, 0.0985, 0.1046, 0.1121, 0.1194, 
					0.1240, 0.1256, 0.1259, 0.1272, 0.1291, 0.1300, 0.1352, 0.1428, 0.1541), #CAVE
}

def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    if dilation==1:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias)
    elif dilation==2:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=dilation)

    else:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=3, bias=bias, dilation=dilation)


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

class CALayer2DFeatsDim(nn.Module):
	def __init__(self, channels, n_feats, reduction=4, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats

		self.pool = nn.AdaptiveAvgPool2d(1)

		self.conv_du = nn.Sequential(
			nn.Conv2d(n_feats, n_feats // reduction, kernel_size=(1, 1), stride=(1, 1)),
			act,
			nn.Conv2d(n_feats // reduction, n_feats, kernel_size=(1, 1), stride=(1, 1)),
			nn.Sigmoid(),
		)

	def forward(self, x):
		# x: [N, F, C, H, W]
		N, F, C, H, W = x.shape
		out = x.transpose(1,2).reshape(N * C, F, H, W) # [N, C, F, H, W]

		attn = self.pool(out)
		attn = self.conv_du(attn)
		out = out * attn

		out = out.reshape(N, C, F, H, W).transpose(1, 2)	# [N, F, C, H, W]
		return out


class CALayer2DFeatsDimWithChansPosEmbed(nn.Module):
	def __init__(self, channels, n_feats, reduction=4, act=nn.ReLU(inplace=True)):
		super().__init__()
		self.channels = channels
		self.n_feats = n_feats

		self.pool = nn.AdaptiveAvgPool2d(1)

		self.conv_du = nn.Sequential(
			nn.Conv2d(n_feats, n_feats // reduction, kernel_size=(1, 1), stride=(1, 1)),
			act,
			nn.Conv2d(n_feats // reduction, n_feats, kernel_size=(1, 1), stride=(1, 1)),
			nn.Sigmoid(),
		)

		self.chan_pos_embed = nn.Parameter(torch.randn(channels, n_feats, 1, 1))	# [C, F, 1, 1]

	def forward(self, x):
		# x: [N, F, C, H, W]
		N, F, C, H, W = x.shape
		out = x.transpose(1,2).reshape(N * C, F, H, W) # [N, C, F, H, W]

		attn = self.pool(out)	# [N*C, F, 1, 1]
		attn = attn + self.chan_pos_embed.repeat(N, 1, 1, 1)
		attn = self.conv_du(attn)
		out = out * attn

		out = out.reshape(N, C, F, H, W).transpose(1, 2)	# [N, F, C, H, W]
		return out


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
		attn  = self.pool(out).reshape(-1, self.channels * self.n_feats)	# [N, C*F]
		attn = self.fc(attn).reshape(-1, self.channels, self.n_feats).unsqueeze(-1).unsqueeze(-1)
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
		attn  = self.pool(x).reshape(-1, self.n_feats * self.channels )	# [N, F*C]
		attn = self.fc(attn).reshape(-1, self.n_feats, self.channels).unsqueeze(-1).unsqueeze(-1)
		out = x * attn
		return out


class DoubleConv2DFeatsDim(nn.Module):
	"""从特定波段，从不同的特征维度，提取空间特征。空间维度共享参数，光谱维度共享参数，特征维度独立参数。"""
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
	"""从所有波段，特定的特征维度，提取空间、光谱维度信息。空间维度共享参数，光谱维度独立参数，特征维度共享参数"""
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
	"""从所有波段，所有的特征维度，提取空间、光谱维度信息。 空间维度共享参数，光谱维度共享参数，特征维度独立参数"""
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
		out = self.conv(x)
		return out

class DoubleConv3dFeatsDim(nn.Module):
	"""从所有波段，所有的特征维度，提取空间、光谱维度信息。空间维度共享参数，光谱维度独立参数，特征维度共享参数"""
	pass


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


