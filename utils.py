from networks import cnn as CNN
from networks import mobilenet_v1

import config as cfg
import torchvision.models as models

def getModel(model_name, pretrained = False):

	if model_name == "CNN":
		model = CNN.Conv2dNet()
	elif model_name =="SqueezeNet":
		model = models.squeezenet1_0(pretrained = cfg.pretrained, num_classes = cfg.class_number)
	elif model_name == "AlexNet":
		model = models.alexnet(pretrained = cfg.pretrained, num_classes = cfg.class_number)
	elif model_name == "MobileNet":
		model = mobilenet_v1.Mobilenet_v1()

	return model.to(cfg.device)

