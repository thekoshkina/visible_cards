import os
import swat
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from collections import defaultdict


class Dataset(torch.utils.data.Dataset):
	def __init__(self, root, labels_path, transform, all_labels):
		self.all_labels = all_labels
		self.data = self.make_data(labels_path)
		self.root = root
		self.data_dict = self.create_dict()
		self.transform = transform
	
	def __len__(self):
		return len(self.data)
	
	def get_labels(self):
		return self.labels
	
	def create_dict(self):
		data_dict = defaultdict(list)
		for index, datum in self.data.iterrows():
			data_dict[datum[' LABEL']].append(os.path.join(self.root, datum['IMAGE_FILENAME']))
		return data_dict
	
	def make_data(self, labels_path):
		data = pd.read_csv(labels_path)
		
		new_rows = []
		for index, datum in data.iterrows():
			new_rows.append({'IMAGE_FILENAME': datum['IMAGE_FILENAME'], ' LABEL': self.all_labels.index(datum[' LABEL'])})
		return pd.DataFrame(new_rows)
	
	def __getitem__(self, index):
		image_path = self.data['IMAGE_FILENAME'][index]
		label = self.data[' LABEL'][index]
		image = Image.open(os.path.join(self.root, image_path))  # only load the grayscale image
		image = self.transform(image)
		return image, label

# dataset = Dataset('data/images', 'data/gicsd_labels.csv' )
