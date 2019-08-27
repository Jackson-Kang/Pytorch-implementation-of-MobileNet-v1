from tqdm import tqdm
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

import data as dt
import config as cfg

from utils import getModel


def train(model_name):

	# train model

	train_loader = dt.loadData(train=True)
	model = getModel(model_name = model_name)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
	scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.lr_decaying_step, gamma = cfg.lr_decaying_value)
	saved_model_path = cfg.log_path + model_name + "_cifar10.pt"

	if os.path.isfile(saved_model_path):
		# load saved checkpoint
		
		checkpoint = torch.load(saved_model_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch']
		loss = checkpoint['loss']
	else:
		start_epoch = 1


	model.train()


	print("\nStart training ", model_name, "...")

	for epoch in range(start_epoch, cfg.epochs + 1):
		for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
			data, target = data.to(cfg.device), target.to(cfg.device)

			if cfg.convert_to_RGB:
				batch_size, channel, width, height = data.size()
				data = data.view(batch_size, channel, width, height).expand(batch_size, cfg.converted_channel, width, height)

			optimizer.zero_grad()

			output=model(data)
			loss = criterion(output, target)

			loss.backward()
			optimizer.step()
		
		scheduler.step()

		if cfg.save_model and (epoch % cfg.save_epoch == 0):
			if not(os.path.isdir(cfg.log_path)):
				os.makedirs(os.path.join(cfg.log_path))
			
			torch.save({
					'epoch':epoch + 1,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'loss': loss
				}, saved_model_path)



		print('\tTrain Epoch: {} / {} \t Loss: {:.6f}\n'.format(epoch, cfg.epochs, loss.item()))
	print("Done!\n\n")



if __name__ == "__main__":

	if not cfg.no_cuda: 
		# use GPU
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		torch.cuda.set_device(int(sys.argv[1]))
		torch.cuda.manual_seed(cfg.gpu_seed)
	else:
		# use CPU
		torch.manual_seed(cfg.cpu_seed)

	for model_name in cfg.model_list:	
		train(model_name = model_name)	

