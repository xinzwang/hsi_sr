import argparse
import models
import torch
from thop import profile


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_chans', default=31, type=int)
	parser.add_argument('--model', default='SSPSR')

	args = parser.parse_args()
	print(args)
	return args

def run(args):
	model = getattr(models, args.model)(channels=args.n_chans, scale_factor=2)
	total = sum([param.nelement() for param in model.parameters()])
	print("Number of parameter: %.2fM" % (total/1e6))

	inputs = torch.randn(1, args.n_chans, 256, 256)

	flops, params = profile(model,(inputs, ))

	print('flops:%.2f' %(flops / 1e9))
	print('params:%.2f' % (params / 1e6))

if __name__=='__main__':
	args = parse_args()
	run(args)