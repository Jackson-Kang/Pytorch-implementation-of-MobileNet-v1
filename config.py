import torch

# CNN model configurations

cnn_kernel_size = 3
cnn_stride = 1
cnn_max_pool = (2, 2)


# All model configurations

pretrained = False
batch_size = 32
test_batch_size = 32

class_number = 10

cpu_seed = 1514780611
gpu_seed = 7053313890570024

lr = 0.001
momentum = 0.9
epochs = 100

model_list1 = ["AlexNet"]
model_list2 = ["SqueezeNet"]
model_list3 = ["MobileNet"]

# data configurations

input_data_channel = 3
convert_to_RGB = False
converted_channel = 3


# CPU-GPU configurations

no_cuda = False
	# -> if true, use cpu
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# path configurations

dataset_path = "./datasets"
MNIST_path = dataset_path + "/MNIST"
CIFAR10_path = dataset_path + "/CIFAR10"
ImageNet_path = dataset_path + "/ImageNet"
train_path = CIFAR10_path + "/train/"
test_path = CIFAR10_path + "/test/"

log_path = "./logs/"



# save configurations

save_model = True
time_record = True
