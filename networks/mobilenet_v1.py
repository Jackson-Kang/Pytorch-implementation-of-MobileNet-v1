import torch
import torch.nn as nn
import torch.nn.functional as f

import config as cfg

class Mobilenet_v1(nn.Module):
	def __init__(self):
		super(Mobilenet_v1, self).__init__()

		self.model = nn.Sequential(

					self.Conv_Bn(in_channels=3, out_channels=32, stride=2),

					self.Depthwise_Separable(in_channels=32, out_channels=64, d_stride=1),

					self.Depthwise_Separable(in_channels=64, out_channels=128, d_stride=2),
					self.Depthwise_Separable(in_channels=128, out_channels=128, d_stride=1),

					self.Depthwise_Separable(in_channels=128, out_channels=256, d_stride=2),
					self.Depthwise_Separable(in_channels=256, out_channels=256, d_stride=1),

					self.Depthwise_Separable(in_channels=256, out_channels=512, d_stride=2),

					self.Depthwise_Separable(in_channels=512, out_channels=512, d_stride=1),
					self.Depthwise_Separable(in_channels=512, out_channels=512, d_stride=1),
					self.Depthwise_Separable(in_channels=512, out_channels=512, d_stride=1),
					self.Depthwise_Separable(in_channels=512, out_channels=512, d_stride=1),
					self.Depthwise_Separable(in_channels=512, out_channels=512, d_stride=1),

					self.Depthwise_Separable(in_channels=512, out_channels=1024, d_stride=2),
					self.Depthwise_Separable(in_channels=1024, out_channels=1024, d_stride=1),
		
					nn.AvgPool2d(7),

		)

		self.fc = nn.Linear(1024, cfg.class_number)


	def forward(self, x):
		x = self.model(x)
		x = x.view(-1, 1024)

		x = self.fc(x)
		x = f.log_softmax(x, dim=1)
		return x


	def Conv_Bn(self, in_channels, out_channels, kernel_size=3, stride=1):
		return nn.Sequential(

				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=False, padding=1),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True)

		)


	def Depthwise_Separable(self, in_channels, out_channels, kernel_size=3, d_stride=1, p_stride=1):
		return nn.Sequential(

				self.__Depthwise(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=d_stride), 
				nn.BatchNorm2d(num_features=in_channels),
				nn.ReLU(inplace=True),

				self.__Pointwise(in_channels=in_channels, out_channels=out_channels, stride=p_stride),
				nn.BatchNorm2d(num_features=out_channels),
				nn.ReLU(inplace=True)
		)



	def __Depthwise(self, in_channels, out_channels, kernel_size, stride=1):
		return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, groups=1, bias=False, padding=1)
	
	def __Pointwise(self, in_channels, out_channels, stride=1):
		return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)
