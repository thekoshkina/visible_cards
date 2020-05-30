from PIL import Image
import argparse
from torchvision import transforms
import torch
from train import Model

parser = argparse.ArgumentParser(description='Process some images.')
parser.add_argument('-image_path', type=str, required=True, help='path to the image for inference')
parser.add_argument('-model_weights', type=str, default='best_checkpointt.pth', help='path to the image for inference')


def predict(args):
	# get only the grayscale image that is stored in the blue channel

	img = Image.open(args.image_path)
	if img.mode == 'RGB':
		print("Only taking blue channel from RGB image")
		img = img.split()[2]
	elif img.mode == 'L':
		print("Provided image is already grayscale")

	
	transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor(),
	                                transforms.Normalize(mean=[0.485], std=[0.229])])
	
	# img = img.split()[2]
	img_t = transform(img)
	batch_t = torch.unsqueeze(img_t, 0)
	
	model = Model()
	
	checkpoint = torch.load(args.model_weights)
	model.load_state_dict(checkpoint['state_dict'])
	
	model.eval()
	out = model(batch_t)
	_, preds = torch.max(out, 1)
	
	print('Result: ', args.all_labels[preds.item()])


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-image_path', type=str, required=True, help='path to the image for inference')
	parser.add_argument('-model_weights', type=str, default='resnet34.pth', help='path to the image for inference')
	parser.add_argument('-all_labels', type=list, default=[' FULL_VISIBILITY ', ' PARTIAL_VISIBILITY ', ' NO_VISIBILITY '], help='The list of labels')
	args = parser.parse_args()
	
	predict(args)
