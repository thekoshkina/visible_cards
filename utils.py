import math
import cv2
import pandas as pd
import random
import os
import numpy as np


def mirror_data(dataframe, root):
	"""
	Creates a mirror reflection for  every image in the dataframe and save new images in the same
	folder and adds the new image into the dataframe
	:param dataframe: dataframe with filenames and labels
	:param root: path to the folder with all the images
	:return:
	"""
	new_entries = []
	for index, datum in dataframe.iterrows():
		img = cv2.imread(os.path.join(root, datum['IMAGE_FILENAME']), cv2.IMREAD_UNCHANGED)
		new_image_name = datum['IMAGE_FILENAME'].split(".")[0] + 'm.png'
		flipped_image = np.fliplr(img)
		cv2.imwrite(os.path.join(root, new_image_name), flipped_image)
		new_entries.append({'IMAGE_FILENAME': new_image_name, ' LABEL': datum[' LABEL']})
	df = pd.DataFrame(new_entries)
	return dataframe.append(df, ignore_index=True)


def flip_data(dataframe, root):
	"""
	Turns upside down every image in the dataframe and save new images in the same
	folder and adds the new image into the dataframe
	:param dataframe: dataframe with filenames and labels
	:param root: path to the folder with all the images
	:return: new dataframe dataframe
	"""
	new_entries = []
	for index, datum in dataframe.iterrows():
		img = cv2.imread(os.path.join(root, datum['IMAGE_FILENAME']), cv2.IMREAD_UNCHANGED)
		new_image_name = datum['IMAGE_FILENAME'].split(".")[0] + 'f.png'
		
		flipped_image = np.flipud(img)
		
		cv2.imwrite(os.path.join(root, new_image_name), flipped_image)
		new_entries.append({'IMAGE_FILENAME': new_image_name, ' LABEL': datum[' LABEL']})
	df = pd.DataFrame(new_entries)
	return dataframe.append(df, ignore_index=True)


def rotate_data(dataframe, root):
	"""
	Rotate every image in the dataframe and save new images in the same
	folder and adds the new image into the dataframe
	:param dataframe: dataframe with filenames and labels
	:param root: path to the folder with all the images
	:return:
	"""
	new_entries = []
	for index, datum in dataframe.iterrows():
		img = cv2.imread(os.path.join(root, datum['IMAGE_FILENAME']), cv2.IMREAD_UNCHANGED)
		new_image_name = datum['IMAGE_FILENAME'].split(".")[0] + 'r.png'
		angle = random.uniform(-45, 45)
		image_rotated_cropped = rotate_resize(img, angle)
		cv2.imwrite(os.path.join(root, new_image_name), image_rotated_cropped)
		new_entries.append({'IMAGE_FILENAME': new_image_name, ' LABEL': datum[' LABEL']})
	df = pd.DataFrame(new_entries)
	return dataframe.append(df, ignore_index=True)


def rotate_resize(img, angle):
	"""
	Rotates the images, crops the black border out and resizes to the shape of the original image

	"""
	image_height, image_width = img.shape
	image_rotated = rotate_image(img, angle)
	image_rotated_cropped = crop_around_center(image_rotated,
	                                           *largest_rotated_rect(image_width, image_height, math.radians(angle)))
	return cv2.resize(image_rotated_cropped, img.shape, interpolation=cv2.INTER_AREA)


def rotate_image(img, angle):
	"""
	Rotate image
	"""
	
	rows, cols = img.shape
	rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
	return cv2.warpAffine(img, rotation_matrix, (rows, cols))


def largest_rotated_rect(w, h, angle):
	"""
	Given a rectangle of size wxh that has been rotated by 'angle' (in
	radians), computes the width and height of the largest possible
	axis-aligned rectangle within the rotated rectangle.

	Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

	Converted to Python by Aaron Snoswell
	"""
	
	quadrant = int(math.floor(angle / (math.pi / 2))) & 3
	sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
	alpha = (sign_alpha % math.pi + math.pi) % math.pi
	
	bb_w = w * math.cos(alpha) + h * math.sin(alpha)
	bb_h = w * math.sin(alpha) + h * math.cos(alpha)
	
	gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
	
	delta = math.pi - alpha - gamma
	
	length = h if (w < h) else w
	
	d = length * math.cos(alpha)
	a = d * math.sin(alpha) / math.sin(delta)
	
	y = a * math.cos(gamma)
	x = y * math.tan(gamma)
	
	return (
		bb_w - 2 * x,
		bb_h - 2 * y
	)


def crop_around_center(image, width, height):
	"""
	Given a NumPy / OpenCV 2 image, crops it to the given width and height,
	around it's centre point
	"""
	
	image_size = (image.shape[1], image.shape[0])
	image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))
	
	if (width > image_size[0]):
		width = image_size[0]
	
	if (height > image_size[1]):
		height = image_size[1]
	
	x1 = int(image_center[0] - width * 0.5)
	x2 = int(image_center[0] + width * 0.5)
	y1 = int(image_center[1] - height * 0.5)
	y2 = int(image_center[1] + height * 0.5)
	
	return image[y1:y2, x1:x2]
