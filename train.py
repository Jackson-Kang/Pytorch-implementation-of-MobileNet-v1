import torch
import torch.nn.functional as f
import torch.optim as optim

from tqdm import tqdm
import os

import data as dt
import config as cfg


from utils import getModel




def train(model_name):

	# train model

	train_loader = dt.loadData(train=True)
	model = getModel(model_name = model_name)

	optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

	model.train()

	print("\nStart training...")

	for epoch in range(1, cfg.epochs + 1):
		for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
			data, target = data.to(cfg.device), target.to(cfg.device)

			optimizer.zero_grad()
			output=model(data)

			loss = f.nll_loss(output, target)
			loss.backward()

			optimizer.step()


		print('\tTrain Epoch: {} / {} \t Loss: {:.6f}\n'.format(epoch, cfg.epochs, loss.item()))
	print("Done!\n\n")



	# save model
	print("Saving model...!")

	if cfg.save_model:		
		if not(os.path.isdir(cfg.log_path)):
			os.makedirs(os.path.join(cfg.log_path))
		torch.save(model.state_dict(), cfg.log_path + model_name + "_mnist.pt")

	print("Done!\n\n")



if __name__ == "__main__":

	torch.manual_seed(cfg.seed)

	train(model_name = "CNN")	
	#train(model_name = "")	

