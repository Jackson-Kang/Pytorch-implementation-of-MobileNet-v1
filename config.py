import torch

# model configurations

batch_size = 32
kernel_size = 5
stride = 1
max_pool = (2, 2)

class_number = 10

lr = 0.001
momentum = 0.9

test_batch_size = 32



# GPU configurations

no_cuda = True
	# -> if true, use cpu
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# path configurations

dataset_path = "./datasets"
MNIST_path = dataset_path + "/MNIST/processed"
train_path = MNIST_path + "/training.pt"
test_path = MNIST_path + "/test.pt"

log_path = "./logs/"



# etc configurations

seed = 1234567
epochs = 2

save_model = True
time_record = True
