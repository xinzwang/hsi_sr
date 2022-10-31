"""
Build Dataset and dataloader
"""

from torch.utils.data.dataloader import DataLoader
from datasets import RemoteDataset, CAVEDataset

def SelectDatasetObject(name):
	if name in ['Pavia', 'PaviaU', 'KSC', 'Indian']:
		return RemoteDataset
	elif name == 'CAVE':
		return CAVEDataset
	else:
		raise Exception('Unknown dataset:', name)

def build_dataset(dataset, path, batch_size=32, scale_factor=2, test_flag=False, cls_path=None):
	datasetObj = SelectDatasetObject(dataset)
	dataset = datasetObj(
		path=path,
		scale_factor=scale_factor, 
		test_flag=test_flag,
		cls_path=cls_path
	)
	dataloader = DataLoader(
		dataset=dataset,
		batch_size=batch_size if not test_flag else 1,
		num_workers=8,
		shuffle= (not test_flag)	# shuffle only train
	)
	return dataset, dataloader


def build_dataset_(name, batch_size=32, N=32, scale_factor=2, test_flag=False):
	path = get_dataset_path(name, dataset_prefix)

	dataset = SRDataset(path, N=N, scale_factor=scale_factor, test_flag=test_flag)
	dataloader = DataLoader(
		dataset=dataset,
		batch_size=batch_size,
		num_workers=8,
		shuffle= (not test_flag),	# shuffle only train
	)
	return dataset, dataloader