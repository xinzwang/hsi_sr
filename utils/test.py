"""
Test api
"""
import os
import cv2
import numpy as np
import imgvision as iv
from tqdm import tqdm
# from skimage.measure import compare_ssim, compare_psnr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from functools import partial

import torch

def test(model, dataloader, device):
	psnr, ssim, sam, mse = [], [], [], []
	model.eval()
	with torch.no_grad():
		for i, (lr, hr) in enumerate(tqdm(dataloader)):
			lr = lr.to(device)

			pred = model(lr)
			assert len(pred)==1, Exception('Test batch_size should be 1, not:%d' %(len(pred)))
			# torch->numpy; 1CHW->HWC; [0, 1]
			hr_ = hr.cpu().numpy()[0].transpose(1,2,0)
			pred_ = pred.cpu().numpy()[0].transpose(1,2,0)
			# eval
			# psnr_, ssim_, sam_ = MSIQA(hr, pred)
			# mse_ = 0
			metric = iv.spectra_metric(pred_, hr_, max_v=1.0)
			psnr_, ssim_, sam_, mse_ = metric.PSNR(), metric.SSIM(), metric.SAM(), metric.MSE()
			psnr.append(psnr_)
			ssim.append(ssim_)
			sam.append(sam_)
			mse.append(mse_)
	psnr, ssim, sam, mse = np.mean(psnr),np.mean(ssim),np.mean(sam),np.mean(mse)
	print('[TEST] psnr:%.5f ssim:%.5f sam:%.5f mse:%.5f' %(psnr, ssim, sam, mse))
	return psnr, ssim, sam, mse

def visual(model, dataloader, img_num=3, save_path='img/', err_gain=10, device=None):
	# create save dir
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	# infer and save
	it = iter(dataloader)
	for i in range(min(img_num, dataloader.__len__())):
		lr, hr = next(it)
		lr = lr.to(device)
		with torch.no_grad():
			pred = model(lr)
		# assert len(pred)==1, Exception('Test batch_size should be 1, not:%d' %(len(pred)))
		# torch->numpy; 1CHW->HWC; [0, 1]
		hr_ = hr.cpu().numpy()[0].transpose(1,2,0)
		pred_ = pred.cpu().numpy()[0].transpose(1,2,0)
		# save err_map gray
		err = np.mean(np.abs(pred_ - hr_), axis=2)
		gray = np.mean(pred_, axis=2)
		cv2.imwrite(save_path + '%d_err.png' %(i), cv2.applyColorMap((err * 255 * err_gain).astype(np.uint8), cv2.COLORMAP_JET))
		cv2.imwrite(save_path + '%d_gray.png'%(i), gray * 255.0)
	return





class Bandwise(object): 
	def __init__(self, index_fn):
			self.index_fn = index_fn

	def __call__(self, X, Y):
			C = X.shape[-3]
			bwindex = []
			for ch in range(C):
					x = torch.squeeze(X[...,ch,:,:].data).cpu().numpy()
					y = torch.squeeze(Y[...,ch,:,:].data).cpu().numpy()
					index = self.index_fn(x, y)
					bwindex.append(index)
			return bwindex


cal_bwpsnr = Bandwise(partial(peak_signal_noise_ratio, data_range=1))
cal_bwssim = Bandwise(structural_similarity)


def cal_sam(X, Y, eps=1e-8):
	X = torch.squeeze(X.data).cpu().numpy()
	Y = torch.squeeze(Y.data).cpu().numpy()
	tmp = (np.sum(X*Y, axis=0) + eps) / (np.sqrt(np.sum(X**2, axis=0)) + eps) / (np.sqrt(np.sum(Y**2, axis=0)) + eps)    
	return np.mean(np.real(np.arccos(tmp)))


def MSIQA(X, Y):
	psnr = np.mean(cal_bwpsnr(X, Y))
	ssim = np.mean(cal_bwssim(X, Y))
	sam = cal_sam(X, Y)
	return psnr, ssim, sam


"""Depreciated"""
def cal_psnr(mse):
	return 10 * np.log10(1 / mse)


def mpsnr(bwmse, verbose=False):
	psnrs = []
	for mse in bwmse:
			cur_psnr = cal_psnr(mse)
			psnrs.append(cur_psnr)
	
	if not verbose:
			return np.mean(psnrs)
	else:
		return np.mean(psnrs), psnrs


if __name__=='__main__':
	hsi = np.random.random([64,64,31])
	gauss = cv2.GaussianBlur(hsi, ksize=(3,3), sigmaX=1,sigmaY=1)

	hsi_tensor = torch.Tensor(hsi).unsqueeze(0)
	gauss_tensor = torch.Tensor(gauss).unsqueeze(0)

	psnr, ssim, sam = MSIQA(hsi_tensor, gauss_tensor)
	print(f'psnr:{psnr} ssim:{ssim} sam:{sam}')

	metric = iv.spectra_metric(gauss, hsi, max_v=1.0)
	psnr, ssim, sam, mse = metric.PSNR(), metric.SSIM(), metric.SAM(), metric.MSE()
	print(f'psnr:{psnr} ssim:{ssim} sam:{sam}')



