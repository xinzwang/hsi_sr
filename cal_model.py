import argparse

import torch

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ckpt', default='/data2/wangxinzhe/codes/github.com/xinzwang.cn/hsi_sr/checkpoints/CAVE/OursV01/block=3_layers=4_feats=16_lr=5e-4/epoch=715_psnr=42.36203_ssim=0.99612ckpt.pt')

	args = parser.parse_args()
	print(args)
	return args

def run(args):
	model = torch.load(args.ckpt)['model']

	print(model)
	total = sum([param.nelement() for param in model.parameters()])
	print("Number of parameter: %.2fM" % (total/1e6))

if __name__=='__main__':
	args = parse_args()
	run(args)