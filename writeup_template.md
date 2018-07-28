# **Behavioral Cloning Writeup** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model06.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a 5 layers convolution neural network with 5x5 and 3x3 filter sizes and depths between 36 and 64.
Acativation includes 7 layers of ELU, and the data is normalized in the model using a Keras lambda layer (code line 70).
```sh
model = Sequential()
model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation= 'relu'))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation= 'relu'))
model.add(ELU())
model.add(Convolution2D(48,5,5, subsample=(2,2), activation= 'relu'))
model.add(ELU())
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(ELU())
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(10))
model.add(ELU())
model.add(Dense(1))
```

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (clone.py lines 74). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 14). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 91).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, plus 0.2 correction to the left steering angle, and subract 0.2 for the right angle, use 'min' and 'max' to avoid left and right steering angle beyond 1.0 or less than -1.0.
```sh
correction = 0.2
center_angle = float(batch_sample[3])
if i ==1:
    angles.append(min(1.0, center_angle + correction))
elif i ==2:
    angles.append(max(-1.0, center_angle - correction))
else:
    angles.append(center_angle)
```

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Lenet， I thought this model might be appropriate because Lenet is classic image trainning model, but car keeps driving in straight and drifting off the road at the corner. Then i introduced [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) model, the car driving bettern in the corner, but car keeps drifting off into that gap.

![drifting off point](https://github.com/StephenZhang945/P3_Behavioral-Cloning/blob/master/examples/drifting%20off.png)

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added 'Dropout(0.5)'.

Then I update original data again by running simulator training mode counter-clockwise and clockwise circle and circle, to make sure at least 4K samples.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 69-89) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 31, 158, 24)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 31, 158, 24)   0           elu_1[0][0]                      
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       dropout_1[0][0]                  
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 14, 77, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       elu_2[0][0]                      
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 5, 37, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       elu_3[0][0]                      
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 3, 35, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       elu_4[0][0]                      
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]                  
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        elu_5[0][0]                      
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         elu_6[0][0]                      
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          elu_7[0][0]                      
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
____________________________________________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover to center lane when car drift to the side of road. These images show what a recovery looks like starting from ... :

![left side](https://github.com/StephenZhang945/P3_Behavioral-Cloning/blob/master/examples/left%20side.png)
![right side](https://github.com/StephenZhang945/P3_Behavioral-Cloning/blob/master/examples/right%20side.png)

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

12780/12688 [==============================] - 76s - loss: 0.0310 - val_loss: 0.0253
Epoch 2/3
12780/12688 [==============================] - 31s - loss: 0.0278 - val_loss: 0.0246
Epoch 3/3
12780/12688 [==============================] - 31s - loss: 0.0272 - val_loss: 0.0263

Epoch 1/3
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
name: GRID K520
major: 3 minor: 0 memoryClockRate (GHz) 0.797
pciBusID 0000:00:03.0
Total memory: 3.94GiB
Free memory: 3.91GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GRID K520, pci bus id: 0000:00:03.0)
12600/12688 [============================>.] - ETA: 0s - loss: 0.0310/home/carnd/anaconda3/envs/carnd-term1/lib/python3.5/site-packages/keras/engine/training.py:1569: UserWarning: Epoch comprised more than `samples_per_epoch` samples, which might affect learning results. Set `samples_per_epoch` correctly to avoid this warning.
  warnings.warn('Epoch comprised more than '

After the collection process, I had 15860 number of data points. I then preprocessed this data by add Cropping = ((70,25),(0,0)).

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by that mini validation loss appeared at 3 epochs about,  I used an adam optimizer so that manually training the learning rate wasn't necessary.
