import argparse
import models


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

if __name__=='__main__':
	args = parse_args()
	run(args)