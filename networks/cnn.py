import torch
import torch.nn as nn
import torch.nn.functional as f

import config as cfg

class Conv2dNet(nn.Module):

	def __init__(self):
		"""
			2D Convolutional Layer with 2 conv layer and 2 fully connected layer.
			
			returns: probability distribution for each classes

			
			* Use 2D ConvNet as experimental baseline, and let's compare with another lightweight model. *
		"""

		super(Conv2dNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, cfg.kernel_size, cfg.stride)
		self.conv2 = nn.Conv2d(20, 50, cfg.kernel_size, cfg.stride)
		self.fc1 = nn.Linear(4 * 4 * 50, 500)
		self.fc2 = nn.Linear(500, cfg.class_number)

	def forward(self, x):
		x = f.relu(self.conv1(x))
		x = f.max_pool2d(x, cfg.max_pool[0], [1])
		x = f.relu(self.conv2(x))
		x = f.max_pool2d(x, cfg.max_pool[0],cfg.max_pool[1])
		x = x.view(-1, 4*4*50)
		x = f.relu(self.fc1(x))
		x = self.fc2(x)

		return f.log_softmax(x, dim = 1)