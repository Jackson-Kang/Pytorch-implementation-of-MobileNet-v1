from torchvision import datasets, transforms

import torch
import os
import config as cfg
import errno

def __downloadData(train):
	"""
		download data from server
			- target path: dataset -> MNIST
	"""

	try:
		if not(os.path.isdir(cfg.dataset_path)):
			os.makedirs(os.path.join(cfg.dataset_path))
		else:

			kwargs = {'num_workers': 1, 'pin_memory': True} if cfg.use_cuda else {}

			if train:

				train_loader = torch.utils.data.DataLoader(
					datasets.CIFAR10(cfg.train_path, train=True, download=True,
						transform=transforms.Compose([
							transforms.Resize(256),
							transforms.RandomCrop(224),
							transforms.RandomHorizontalFlip(),
							transforms.ToTensor(),
							transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
						])
					),batch_size=cfg.batch_size, shuffle=True, **kwargs)
				return train_loader			

			else:
				test_loader = torch.utils.data.DataLoader(
					datasets.CIFAR10(cfg.test_path, train=False, download=True, 
						transform=transforms.Compose([
							transforms.Resize(224),
							transforms.ToTensor(),
							transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
						])
					), batch_size=cfg.test_batch_size, shuffle=True, **kwargs)
				print(test_loader)	
				return test_loader

	except OSError as e:
		if e.errno != errno.EEXIST:
			print("Failed to create directory!!!!!")
			exit(-1)


def loadData(train=True):
	"""
		load data from downloaded datasets

	"""

	return __downloadData(train)



