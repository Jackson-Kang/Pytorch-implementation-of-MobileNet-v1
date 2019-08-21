from networks import cnn as CNN

import config as cfg

def getModel(model_name):

	if model_name == "CNN":
		model = CNN.Conv2dNet().to(cfg.device)
	elif model_name =="DepthwiseSeparable":
		model = None

	return model

