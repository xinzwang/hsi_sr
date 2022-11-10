"""
Visual the predict error along channel wise.
"""
import os
import cv2
import time
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter



import sys
sys.path.append('../../')

from utils.logger import create_logger
from utils.dataset import build_dataset
from utils.test import test, visual
from utils.seed import set_seed
from utils.core import SRCore

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='CAVE', choices=['ICVL', 'CAVE', 'Pavia', 'Salinas','PaviaU', 'KSC', 'Indian'])
	parser.add_argument('--scale_factor', default=2, type=int)
	parser.add_argument('--batch_size', default=16, type=int)
	parser.add_argument('--device', default='cuda:1')
	parser.add_argument('--parallel', default=False)
	parser.add_argument('--device_ids', default=['cuda:5', 'cuda:6', 'cuda:7'])

	parser.add_argument('--model', default='SSPSR')
	parser.add_argument('--ckpt', default='/data2/wangxinzhe/codes/github.com/xinzwang.cn/hsi_sr/checkpoints/CAVE/SSPSR/2022-10-31_14:44:37/epoch=3890_psnr=38.72785_ssim=0.99173ckpt.pt')

	# Pavia
	# parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/Pavia/sr/train_x420_y230_N256.npy')
	# parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/Pavia/sr/test_x420_y230_N256.npy')
	# Salinas
	# parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/Salinas/train_x192_45_N128.npy')
	# parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/Salinas/test_x192_45_N128.npy')
	# CAVE
	parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/CAVE/train.npy')
	parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/CAVE/test.npy')
	# ICVL
	# parser.add_argument('--train_path', default='/data2/wangxinzhe/codes/datasets/ICVL/train/')
	# parser.add_argument('--test_path', default='/data2/wangxinzhe/codes/datasets/ICVL/test/')

	args = parser.parse_args()
	print(args)
	return args


def run(args):
	save_path = 'img/%s/%s/' %(args.dataset, args.model)
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	# device
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	# dataset
	dataset, dataloader = build_dataset(
		dataset=args.dataset, 
		path=args.train_path, 
		batch_size=args.batch_size, 
		scale_factor=args.scale_factor, 
		test_flag=False)
	test_dataset, test_dataloader = build_dataset(
		dataset=args.dataset, 
		path=args.test_path, 
		batch_size=1, 
		scale_factor=args.scale_factor, 
		test_flag=True)
	
	# core
	core = SRCore(batch_log=10)
	core.inject_device(device)
	core.load_ckpt(args.ckpt)
	
	# predict train
	train_err = core.predict(dataloader)
	print(train_err.shape)
	# print('train_loss:%.5f' % (train_loss))
	plt.figure(dpi=200)
	for x in train_err:
		plt.plot(range(len(x)), x, linewidth=1)
	err_avg = np.mean(train_err, axis=0)
	plt.plot(range(len(err_avg)), err_avg, color = "r", linewidth=3)
	plt.savefig(save_path + 'train_channel_err.png')

	# predict test
	test_err = core.predict(test_dataloader)
	print(test_err.shape)
	# print('test_loss:%.5f'%(test_loss))
	plt.figure(dpi=200)
	for x in test_err:
		plt.plot(range(len(x)), x, linewidth=1)
	err_avg = np.mean(test_err, axis=0)
	plt.plot(range(len(err_avg)), err_avg, color = "r", linewidth=3)
	plt.savefig(save_path + 'test_channel_err.png')


if __name__=='__main__':
	args = parse_args()
	run(args)