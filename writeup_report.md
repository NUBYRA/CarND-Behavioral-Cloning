# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

Poject includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model implemented is the [COMMA.ai Model](https://arxiv.org/abs/1608.01230)

The model consist of one 8 by 8 filter, and two 5 by 5 filters, with depth ranging from 16 to 64. The model includes ELU layers to introduce nonlinearity (code line 20), and the data is normalized within the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The sample data was randomly divided into training (80%) and validation (20%) sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. No addiional changes were made to the COMMA.ai model. As can be seen from the training and validation loss, model does not overfit for any epohcs and loss is very small.  

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data used was the one provided by Udacity, no additioal data was collected, rather relied healivy on data augmentation. Images from center, left and right camera was utilized to also increase sample size. 

For details about how training data was created, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

After data augmentation and preprocessing (see next section), both the NVIDIA and COMMA.ai model were evaluated. Eventhough both models perform well, went for COMMA.ai because wanted to train on a lighter model for faster output.

In order to gauge how well the model was working, image and steering angle data were split into a training and validation set. The model was run for 3 ephocs with 20000 samples per epoch and batch size of 64. Model had low mean squared error on the training and validation set. This implied that the model was not overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. The vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes.

|Layer (type)             									 |Output Shape| 
|------------------------------------------------------------|------------|
|Lambda    				  									 |(64, 64, 3) | 
|Convolution (Filter: 8 X 8, Stride: 4 X 4, Activation: ELU) |(16, 16, 16)|  
|Convolution (Filter: 5 X 5, Stride: 2 X 2, Activation: ELU) |(8, 8, 32)  |  
|Convolution (Filter: 5 X 5, Stride: 2 X 2, Activation: ELU) |(4, 4, 64)  |          
|Flatten         											 |1024        |  
|Dropout (prob = .2)        								 |1024        |  
|Activation (ELU)          									 |1024        | 
|Fully Connected (Activation: ELU)        				     |512         |    
|Dropout (prob = .5)          								 |512 	      |    
|Activation (ELU)          									 |512         |  
|Fully Connected         				     				 |1 (Steering)|  
           

#### 3. Creation of the Training Set & Training Process

Only sample data provided by Udacity was used, no additional data was collected. The sample data is only collected from Track 1. To keep vehicle on track and generalize to other tracks, such as Track 2, relied heavily on data augmentation. 

To augment the data set, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 4. Test Demonstration of Track 1 and Track 2 

### Track 1

<script src="http://vjs.zencdn.net/4.0/video.js"></script>

<video id="Track1" class="video-js vjs-default-skin" controls
preload="auto" width="320" height="160" poster="./examples/track1.png"
data-setup="{}">
<source src="run1.mp4" type='video/mp4'>
</video>

