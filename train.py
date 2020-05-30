import torch
from torchvision import models
from dataset import Dataset
from torchvision import transforms
from torch.autograd import Variable
import argparse


class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		
		self.resnet = models.resnet34(pretrained=False)
		self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
		                                    bias=False)  # change to single channel input
		
		self.resnet = load_weights_single_channel(self.resnet,
		                                          "https://download.pytorch.org/models/resnet34-333f7ec4.pth")
		
		for param in self.resnet.parameters():
			param.requires_grad = True
		
		self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 3)  # change to 3 class classification
		
		self.softmax = torch.nn.Softmax()
	
	def forward(self, x):
		x = self.resnet(x)
		x = self.softmax(x)
		return x


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def set_parameter_requires_grad(model, feature_extracting):
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False


def load_weights_single_channel(model, url):
	state_dict = torch.utils.model_zoo.load_url(url)
	conv1_weight = state_dict['conv1.weight']
	state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
	model.load_state_dict(state_dict)
	return model


def main(args):
	transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor(), transforms.Normalize(mean=[0.485], std=[0.229])])
	
	train_data = Dataset(args.root, args.annotations_train, transform, args.all_labels)
	val_data = Dataset(args.root, args.annotations_val, transform, args.all_labels)
	dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
	val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
	
	model = Model()
	weight = torch.tensor([1.0, 2.0, 2.0])
	criterion = torch.nn.CrossEntropyLoss()
	optimiser = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=7, gamma=0.1)
	
	best_acc = 0
	for epoch in range(args.epochs):
		train_loss, train_accuracy = train(model, criterion, optimiser, lr_scheduler, dataloader, epoch, args.epochs)
		
		validation_loss, acc = validation(model, criterion, optimiser, lr_scheduler, val_dataloader, epoch, args.epochs)
		if acc > best_acc:
			best_acc = acc
			state = {
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'optimizer': optimiser.state_dict(),
				'best_acc': best_acc
			}
			torch.save(state, args.checkpoint)


def train(model, criterion, optimiser, scheduler, dataloader, epoch, epochs):
	losses = AverageMeter()
	accuracies = AverageMeter()
	
	model.train()  # Set model to training mode
	for batch, (inputs, labels) in enumerate(dataloader):
		
		inputs = Variable(inputs)
		labels = Variable(labels)
		
		# print(labels)
		outputs = model(inputs)
		# _, preds = torch.max(outputs, 1)
		loss = criterion(outputs, labels)
		
		# calculate accuracies
		acc = calculate_accuracy(outputs, labels)
		
		# statistics
		losses.update(loss.item(), inputs.size(0))
		accuracies.update(acc, inputs.size(0))
		
		optimiser.zero_grad()
		loss.backward()
		optimiser.step()
		scheduler.step()
		
		if batch % 10 == 0:
			print('Epoch {}/{}:[{}]/[{}] Loss: {:.4f} Acc: {:.4f}'.format(epoch, epochs, batch, len(dataloader),
			                                                              losses.avg, accuracies.avg))
	
	return losses.avg, accuracies.avg


def validation(model, criterion, optimiser, scheduler, dataloader, epoch, epochs):
	losses = AverageMeter()
	accuracies = AverageMeter()
	model.eval()  # Set model to validation mode
	for batch, (inputs, labels) in enumerate(dataloader):
		outputs = model(inputs)
		# _, preds = torch.max(outputs, 1)
		loss = criterion(outputs, labels)
		
		# calculate accuracies
		acc = calculate_accuracy(outputs, labels)
		# precision = calculate_precision(outputs, labels)  #
		# recall = calculate_recall(outputs, labels)
		
		losses.update(loss.item(), inputs.size(0))
		accuracies.update(acc, inputs.size(0))
		if batch % 10 == 0:
			print('Val epoch {}/{}:[{}]/[{}] Loss: {:.4f} Acc: {:.4f}'.format(epoch, epochs, batch, len(dataloader), losses.avg, accuracies.avg))
	
	return losses.avg, accuracies.avg


def calculate_accuracy(outputs, targets):
	batch_size = targets.size(0)
	
	_, pred = outputs.topk(1, 1, True)
	pred = pred.t()
	correct = pred.eq(targets.view(1, -1))
	n_correct_elems = correct.float().sum().item()
	
	return n_correct_elems / batch_size


if __name__ == "__main__":
	all_labels = [' FULL_VISIBILITY ', ' PARTIAL_VISIBILITY ', ' NO_VISIBILITY ']
	
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-epochs', type=int, default=20, help='number of epoch for training')
	parser.add_argument('-batch_size', type=int, default=16, help='number of epoch for training')
	parser.add_argument('-checkpoint', type=str, default='resnet34.pth', help='path where to save checkpoint during training')
	parser.add_argument('-root', type=str, default='data/images_grayscale', help='path to the folder with grayscale images')
	parser.add_argument('-annotations_train', type=str, default='data/gicsd_labels_train.csv', help='path to the folder with grayscale images')
	parser.add_argument('-annotations_val', type=str, default='data/gicsd_labels_val.csv', help='path to the folder with grayscale images')
	parser.add_argument('-all_labels', type=list, default=[' FULL_VISIBILITY ', ' PARTIAL_VISIBILITY ', ' NO_VISIBILITY '], help='The list of labels')
	args = parser.parse_args()
	main(args)
	print('Done.')
