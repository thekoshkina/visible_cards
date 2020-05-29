***


# Classifying the visibility of ID cards in photos

The folder images inside data contains several different types of ID documents taken in different conditions and backgrounds. The goal is to use the images stored in this folder and to design an algorithm that identifies the visibility of the card on the photo (FULL_VISIBILITY, PARTIAL_VISIBILITY, NO_VISIBILITY).

## Data

Inside the data folder you can find the following:

### 1) Folder images
A folder containing the challenge images.

### 2) gicsd_labels.csv
A CSV file mapping each challenge image with its correct label.
	- **IMAGE_FILENAME**: The filename of each image.
	- **LABEL**: The label of each image, which can be one of these values: FULL_VISIBILITY, PARTIAL_VISIBILITY or NO_VISIBILITY. 

## Dependencies
The dependencies can be installed by running: 
```
pip3 install -r requirements.txt
```
## Run Instructions
```
train.py -label_path [path_to_annotation_file] -root_path [path_to_image_folder]
```
```
predict.py -image_path [path_to_image] 
```

## Approach
The main challenge of this task is to build a model capable of classifying the visibility of ID cards given the small dataset. In addition to the small size, the dataset is also unbalanced with the smallest class NO_VISIBILITY having 20 times fewer members than the most extensive class FULL_VISIBILITY. Finally, because the data was collected with a faulty sensor and red and green channel only have Gaussian noise with a grayscale image.   
This last restriction means that we can't take full advantage of multiple models pretrained from image classification on ImageNet or from ID card segmentation pretrained of full MD-500 dataset. 

All this considering I start by oversampling the less prevalent classes and augmenting the images to create more samples. I save all the images in the folder and create three annotation files: test, train and validation. The images in the test set are not augmented at all to make sure that the model performance on the test set is the same as production data.

The classifier for the images is based on a pre-trained ResNet 34 model from the PyTorch model zoo.  ResNet is a 34-layer-deep convolutional network that takes care of the vanishing gradient problem with the use of the residual module.  
Since the model has been trained on RGB image with 3 channels, I modified the first layer to make it work with a single-channel grayscale image. I have also modified the fully connected layer to differentiate between 3 classes and added a softmax at the end.

I have chosen PyTorch as a platform for this project and used Dataloader from it to load the images during model training.

Another way to get around the channel issue is to pass a grayscale image in all 3 channels to the pretrained model. I didn't follow this path because the task in the challenge specified that the model has to take a single channel input. 

On my side, the challenge is made harder by the fact that I don't have access to a GPU with CUDA support, as I'm doing this technical challenge on my MacBook. This means that I can't train the model entirely and have to submit an undertrained model, the accuracy of which is not as good as it could've been if I had enough computing power. 

## Future Work

Depending on the results of the fully trained model, the architecture of the model might be changed accordingly: 
If the model is underfitting, we should try a deeper architecture, and if it's overfitting, we should try decreasing the depth of the model or collecting more data. 

Other important tasks that I didn't have time for:  
- Feature visualisation 
- Feature attribution analysis - see the importance of every feature for specific input
- Error analysis - seeing which images the model got wrong
- Saliency map analysis  -  highlight which regions of an image the model was looking at


