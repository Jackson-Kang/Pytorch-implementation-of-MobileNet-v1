from tqdm import tqdm

import torch
import torch.nn.functional as f

from utils import getModel

import config as cfg
import data as dt



def test(model_name):

	model = getModel(model_name = model_name)
	test_loader = dt.loadData(train=False)

	test_loss = 0
	correct = 0

	print("\nLoad saved model...")
	model.load_state_dict(torch.load(cfg.log_path + model_name + "_mnist.pt"))
	model.eval()
	print("Done..!")

	print("\nStart to test...")
	with torch.no_grad():
		for data, target in tqdm(test_loader):
			data, target = data.to(cfg.device), target.to(cfg.device)
			output = model(data)
			test_loss += f.nll_loss(output, target, reduction='sum').item() # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)

	print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n\n'.format(
        		test_loss, correct, len(test_loader.dataset),
        		100. * correct / len(test_loader.dataset))
	)



if __name__ == "__main__":
	test(model_name="CNN")

