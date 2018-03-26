# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/14.jpg "Traffic Sign 1" 
[image5]: ./test_images/23.jpg "Traffic Sign 2" 
[image6]: ./test_images/25.jpg "Traffic Sign 3"
[image7]: ./test_images/3.jpg "Traffic Sign 4"
[image8]: ./test_images/36.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/amita-kapoor/Udacity-SDC-Traffic-Signs/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
> There are 34799 examples in the training dataset. Each with a 32x32x3 RGB image
* The size of the validation set is ?
> 4410 images in validation set
* The size of test set is ?
> The test data set has 12630 images
* The shape of a traffic sign image is ?
> The shape of the image is 32x32x3
* The number of unique classes/labels in the data set is ?
> The number of unique classes in the dataset are 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set, different traffic sign images along with the sign name. 
![alt text](/test_images/original_dataset.png)

Here you can also see the bar chart showing how the data is distributed among different classes.
![bar_chart](/test_images/bar_chart1.png)


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to do three basic preprocessing steps:
* Converted the image to grayscale, this was done because in my opinion the color of traffic signs does not convey any information and so there is no point in making the networks learn it and later get confused. especially since with time colors change.
* Increased the intensity of individual pixels by multiplying it by a constant number, now this actually is more than pure pre-processing, it is like increasing the 'brightness' of the image. I randomly increased the brightness. This was to ensure a more realistic dataset, since depending upon the ambient light the self driving car (SDC) may see different intensities of the same traffic sign.
* Lastly I normalized the data, for this I used the standard: `(pixel - 128)/ 128` formula. Normalization helps in restricting the input and hence output of the neurons to small numbers. This ensures that the gardients do not explode.
 
I performed all three in one step by declaring an object X_PreProcess of the class ImagePreProcess which I defined as: 
'''
class ImagePreProcess(object):
    def __init__(self, gray_scale=True,normalize=128.0, scale_factor=5):
        self.normalize_factor = normalize
        self.gray = gray_scale
        self.scale_factor = scale_factor
    
    def normalize(self, Image):
        result = (Image - self.normalize_factor)/self.normalize_factor  # Numpy Broadcast will take care of the matrix sizes
        return result
    
    def convert_to_gray(self,Image):
        return np.sum(Image/3, axis=3, keepdims=True) 
    
    def scaled(self,Image):
        result = Image + np.random.random()*self.scale_factor
        return result
    
    def fit(self,X):
        if self.gray:
            X = self.convert_to_gray(X)
            print(X.shape)
        X = self.scaled(X)
        X = self.normalize(X)
        return X
'''
Here is an example of a traffic sign images after preprocessing

![alt text](/test_images/preprocess_dataset.png)

Intially I experimented without data Augmenation, but there was only 92% accuracy from the LeNet model. Therefore I decided to augment the dataset. I performed only two augmentation tasks, viz: shift and rotate. As the car moves through the road, there is a chance that traffic signs shift there position and may appear rotated. This is not an exhaustive list, many more things can happen, but for now I restricted myself to these two.

Below you can see the augmented images

![alt text](/test_images/augment_dataset.png)

Adding the augmented dataset to training dataset I now had 64799 images. Here you can see the bar chart of the distribution of the augmented dataset.
![alt_text](/test_images/bar_chart2.png)


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Both the LeNet model and mofdified MultiScale CNN have been explored in the notebook.

The Architecture of LeNet used is:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScale image   							| 
| Convolution Layer 1   |5x5 kernel, stride=1, padding VALID, filters 6, outputs 28x28x6 	|
| ReLu					|												|
| Max pooling	      	| 2x2 kernel, stride = 1,  outputs 14x14x6 		|
| Convolution Layer 2	| 5x5 kernel, stride = 1, padding VALID, filters 6,, outputs 10x10x16 |
| ReLu					|												|
| Max pooling	      	| 2x2 kernel, stride = 1,  outputs 5x5x16 		|
| Falatten              | Input= 5x5x16 Output = 400                    |
| Fully connected	1	| Input = 400, Output= 120        				|
| ReLu  				|        								    	|
| DropOut               | 0.7                                           |
| Fully connected	2	| Input = 120, Output= 84        				|
| ReLu  				|        								    	|
| DropOut               | 0.7                                           |
| Fully connected	2	| Input = 84, Output= 43        				|
| Linear 				|        								    	|
|						|												|
|						|												|
 
And in the modified MultiScale CNN the output of the second convolutional layer is fed alongwith the output of the third convolutional layer. From the paper the architecture of this model is:
![Lenet2](/test_images/lenet_modified.png)

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an used Adam Optimizer with a learning rate of 0.001. (Initially due to typo error I forgot one zero and wasted 3 hours finding why my network is not learning :smiley: ). The batch size of 100 was chosen. And total epochs of 100. To ensure that due to overfitting network peformance does not degrade, I am saving the model with highest validation accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?

> Model 1 is based on the LeNet model as proposed by Yann LeCun, and Model 2 is based on the MultiScale CNN as described in [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) I chose them because both are time-tested architectures and for traffic signs there should not be a need of more complicated network.
To ensure that due to overfitting network peformance does not degrade, I am saving the model with highest validation accuracy.
The last fully connected layer in both is a linear layer instead of Softmax, I observed that with Linear it is giving faster and better results, while with the Softmax it was very slowly converging and even after 100 epoch reached only 75%

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

My final model results were:

| Data set            |      Model 1  Accuracy  | Model 2  Accuracy    |
|:-------------------:|:-----------------------:|:--------------------:| 
| Training set        |  98.815                 |  99.846%             |
| Validation set      |  94.739%                |  95.782%             |
| Test set            |  93.45%                 |  93.444%             |

In my intial attempts LeNet was showing overfitting, to take care of that I added dropout layers. I had to tune droput layers `keep_prob` parameter, I tried various values ranging from 0.2-0.7, finally chose 0.7.
If an iterative approach was chosen:
 Each convolution layer extracts a feature from the data, we can see from the feature map that the first Convolutional layer is extracting the basic shape of the traffic sign, and the second Convolutional layer some abstract information.
 ![original_sign](/test_images/original_sign.png)
 ![conv1_feature_map](/test_images/conv1_feature_map.png)
 ![conv2_feature_map](/test_images/conv2_feature_map.png)



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 

This image is the image of stop sign. While the sign itself is very clear and should not be difficult to classify.

![alt text][image5] 

This image is of Slippery road, the image has lot of watermarks on it, and might be difficult to classify on account of it.

![alt text][image6] 

This is the image of Road Work, the sign has few signs of wear and tear, but I think network can classify it easily.

![alt text][image7] 

This is the image of speed limit, unlike most training samples the board on which the sign id rectangular not round, this may cause mis-classification.

![alt text][image8]

This is the image of Go straight or right sign. It is quite clear and should be easy to guess.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop   		        | Stop		    	            				| 
| Slipppery Road   		| Slippery Road		    	     				| 
| Road work     		| Road work 									|
| Speed limit (60Km/h)	| Speed limit (60Km/h)							|
| Go Straight or right	| Go Straight or right		    				|


The model was able to correctly all the images, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.74%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on these web scrapped images is on the cell 32nd. Both models gave 100% accuracy on these images, which is surprising and satisfying.

For the first image, the model is relatively sure that this is a **stop sign** (probability of 0.99), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction (Model 2)     					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99   		        | Stop		    	            				| 
| 3.32e-6        		| Road Work		    	     		    		| 
| 2.21e-8       		| Keep Right			        				|
| 2.01e-12          	| Yield                              			|
| 7.05e-13          	| Speed limit (50km/h)		    				|


For the second image, the model is relatively sure that this is a **slippery road sign** (probability of 1.00), and the image does contain a slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction (Model 2)     					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00   		        | Slippery Road    	            				| 
| 4.0e-22        		| Bicycles crossing		    	   				| 
| 1.0e-23       		| Dangerous curve to the right					|
| 1.1e-26           	| No passing for vehicles over 3.5 metric tons  |
| 6.6e-35          	    | Keep right		            				|


For the third image, the model is relatively sure that this is a **road work sign** (probability of 1.00), and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction (Model 2)     					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00   		        | Road work     	            				| 
| 1.8e-25        		| Road narrows on the right    	   				| 
| 4.9e-28       		| Go straight or left 			        		|
| 3.9e-34           	| Bumpy road                                    |
| 5.9e-35          	    | Bicycles crossing 		         			|

For the fourth image, the model is relatively sure that this is a **speed limit (60Km/h)** sign (probability of 1.00), and the image does contain a speed limit (60Km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction (Model 2)     					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00   		        | Speed limit (60Km/h)            				| 
| 1.8e-11        		| Speed limit (50km/h)	    	   				| 
| 2.6e-24       		| Speed limit (80km/h)		         			|
| 2.2e-26           	| No passing for vehicles over 3.5 metric tons  |
| 0.0e-00          	    | Speed limit (20km/h)            				|

For the fifth image, the model is relatively sure that this is a **Go straight or right sign** (probability of 1.00), and the image does contain a Go straight or rightk sign. The top five soft max probabilities were

| Probability         	|     Prediction (Model 2)     					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00   		        | Go Straight or right            				| 
| 3.2e-17        		| Turn left ahead		    	   				| 
| 6.0e-26       		| Roundabout mandatory      					|
| 5.4e-28           	| Dangerous curve to the right                  |
| 1.8e-28          	    | Go straight or left		            		|

Below you can see the input image and the top 3 guesses in image format
![top_3](/test_images/top_3.png)


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
Input image
![input](/test_images/input.png)
The first convolutional layer has six filters, we can see that the first convolutional layers has learned the basic shapes in the input image like triangle, circle, and even numbers.
![conv1](/test_images/conv1.png)
The second convolutional layer has sixteen filters, we can see that this layer has learned a complex abstract information about the input.
![conv2](/test_images/conv2.png)
