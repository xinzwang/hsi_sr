import os
import cv2
from tqdm import tqdm
import numpy as np
import argparse
from tqdm import tqdm

from sklearn.cluster import KMeans

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='CAVE')
	parser.add_argument('--n_clusters', default=19, type=int)
	parser.add_argument('--use_test', default=True, type=bool)
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

def down_sample(x, scale_factor=2, kernel_size=(9,9), sigma=3):
	out = cv2.GaussianBlur(x, ksize=kernel_size, sigmaX=sigma,sigmaY=sigma)
	out = cv2.resize(out, (0,0), fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_CUBIC)
	return out

def img2vec(data):
	N, H,W,C = data.shape
	out = data.reshape(N*H*W, C)
	return out

def vec2img(data, shape):
	N, H,W, C = shape
	out = data.reshape(N,H,W)
	return out

def cluster(train_data, test_data, n_clusters=9, use_test=False):
	# image 2 vector
	fit_train = img2vec(train_data)
	fit_test = img2vec(test_data)

	# fit and predict
	model = KMeans(n_clusters=n_clusters)
	if use_test:
		N_train = fit_train.shape[0]
		N_test = fit_test.shape[0]
		print('N_train:', N_train)
		print('N_test:', N_test)
		fit_data = np.concatenate((fit_train, fit_test), axis=0)
		# fit all data
		pred = model.fit_predict(fit_data)
		pred_train = pred[0:N_train]
		pred_test = pred[N_train:]
	else:
		# fit train data
		pred_train = model.fit_predict(fit_train)
		# only predict test data
		pred_test = model.predict(fit_test)

	# vector 2 image
	fit_train = vec2img(pred_train, train_data.shape)
	fit_test = vec2img(pred_test, test_data.shape)

	return fit_train, fit_test


def DownsampleAndCluster(train_path, test_path, n_clusters=9, use_test=False):
	"""对下采样的图片进行聚类"""
	# downsample all images
	train_data = []
	for i in tqdm(range(train_data_.shape[0])):
		train_data.append(down_sample(train_data_[i]))
	train_data = np.array(train_data)
	print('train shape:', train_data.shape)
	
	test_data = []
	for i in tqdm(range(test_data_.shape[0])):
		test_data.append(down_sample(test_data_[i]))
	test_data = np.array(test_data)
	print('test shape:', test_data.shape)

	# cluster
	fit_train, fit_test = cluster(train_data, test_data, n_clusters=n_clusters, use_test=use_test)

	return fit_train, fit_test


def ClusterAndDownsample(train_data, test_data, n_clusters=9, use_test=False):
	"""先聚类再下采样"""
	# cluster
	fit_train, fit_test = cluster(train_data, test_data, n_clusters=n_clusters, use_test=use_test)

	# downsample cls map using nearest
	train_res = []
	for i in tqdm(range(fit_train.shape[0])):
		img = fit_train[i]
		img = cv2.resize(img, (0,0), fx=1/2, fy=1/2, interpolation=cv2.INTER_NEAREST)
		train_res.append(img)
	train_res = np.array(train_res)
	print('train shape:', train_res.shape)
	
	test_res = []
	for i in tqdm(range(fit_test.shape[0])):
		img = fit_test[i]
		img = cv2.resize(img, (0,0), fx=1/2, fy=1/2, interpolation=cv2.INTER_NEAREST)
		test_res.append(img)
	test_res = np.array(test_res)
	print('test shape:', test_res.shape)

	return train_res, test_res



def run(args):
	train_data = np.load(args.train_path)	# NHWC
	test_data = np.load(args.test_path)		# NHWC

	# predict
	# fit_train, fit_test = DownsampleAndCluster(train_data, test_data, n_clusters=args.n_clusters, use_test=args.use_test)

	fit_train, fit_test = ClusterAndDownsample(train_data, test_data, n_clusters=args.n_clusters, use_test=args.use_test)

	print('train pred shape:', fit_train.shape)
	print('test pred shape:', fit_test.shape)

	# save results
	train_name = args.train_path.split('.')[0]
	test_name = args.test_path.split('.')[0]
	
	np.save(train_name + '_kmeans_hr_%d_%d' %(args.n_clusters, args.use_test), fit_train)
	np.save(test_name + '_kmeans_hr_%d_%d' %(args.n_clusters, args.use_test), fit_test)

	# visual some images
	K = 15
	path = 'img/k-means/%s/hr_n_clusters=%d_usetest=%d/'%(args.dataset,args.n_clusters, args.use_test)
	if not os.path.exists(path):
		os.makedirs(path)
		
	for i in range(min(len(fit_train), 15)):
		train_img = fit_train[i]
		cv2.imwrite(path + 'train_%d_cls.png'%(i), cv2.applyColorMap((train_img / args.n_clusters *  255).astype(np.uint8), cv2.COLORMAP_JET))
		

	for i in range(min(len(fit_test), 15)):
		test_img = fit_test[i]
		cv2.imwrite(path + 'test_%d_cls.png'%(i), cv2.applyColorMap((test_img / args.n_clusters *  255).astype(np.uint8), cv2.COLORMAP_JET))
		
if __name__=='__main__':
	args = parse_args()
	run(args)