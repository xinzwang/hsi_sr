import os
import cv2
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from utils.logger import create_logger
from utils.dataset import build_dataset
from utils.test import test, visual
from utils.seed import set_seed
from utils.core import SRCore

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='ICVL', choices=['ICVL', 'CAVE', 'Pavia', 'Salinas','PaviaU', 'KSC', 'Indian'])
	parser.add_argument('--scale_factor', default=2, type=int)
	parser.add_argument('--batch_size', default=16, type=int)
	parser.add_argument('--epoch', default=10001)
	parser.add_argument('--device', default='cuda:2')

	parser.add_argument('--ckpt', default='/data2/wangxinzhe/codes/github.com/xinzwang.cn/hsi_sr/checkpoints/ICVL/SSPSR/2022-11-01_14:51:01/epoch=235_psnr=53.70649_ssim=0.99852ckpt.pt')
	# Pavia
	# parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/Pavia/sr/train_x420_y230_N256.npy')
	# parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/Pavia/sr/test_x420_y230_N256.npy')
	# Salinas
	# parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/Salinas/train_x192_45_N128.npy')
	# parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/Salinas/test_x192_45_N128.npy')
	# CAVE
	# parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/CAVE/train.npy')
	# parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/CAVE/test.npy')
	# ICVL
	# parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/ICVL/train/')
	parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/ICVL/test/')

	args = parser.parse_args()
	print(args)
	return args

def run(args):
	device=torch.device(args.device)

	test_dataset, test_dataloader = build_dataset(
		dataset=args.dataset, 
		path=args.test_path, 
		batch_size=16, 
		scale_factor=args.scale_factor, 
		test_flag=True)

	ckpt = torch.load(args.ckpt, map_location=device)
	model = ckpt['model']

	print(test(model, test_dataloader, device))


if __name__=='__main__':
	args = parse_args()
	run(args)