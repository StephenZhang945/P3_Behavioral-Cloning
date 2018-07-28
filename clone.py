import csv
import cv2
import numpy as np
import random
import sklearn
from sklearn.model_selection import train_test_split

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size = 30):
    num_samples = len(samples)
    correction = 0.2
    while True:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    sample_path = batch_sample[i]
                    sample_name = sample_path.split('/')[-1]
                    name = './data/IMG/'+sample_name
                    image1 = cv2.imread(name)
                    image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
                   # image1 = np.array(image1, dtype = np.float64)
                   # random_bright = .25+np.random.uniform()
                   # image1[:,:,2] = image1[:,:,2]*random_bright
                   # image1[:,:,2][image1[:,:,2]>255]  = 255
                   # image1 = np.array(image1, dtype = np.uint8)
                   # image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
                    center_angle = float(batch_sample[3])
                    images.append(image1)
                    if i ==1:
                        angles.append(min(1.0, center_angle + correction))
                    elif i ==2:
                        angles.append(max(-1.0, center_angle - correction))
                    else:
                        angles.append(center_angle)
        
        
            aug_images, aug_angles = [], []
            for image, angle in zip(images, angles):
                aug_images.append(image)
                aug_angles.append(angle)
                aug_images.append(cv2.flip(image, 1))
                aug_angles.append(angle*-1.0)
 
            X_train = np.array(aug_images)
            y_train = np.array(aug_angles)
            yield sklearn.utils.shuffle(X_train, y_train)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

train_generator = generator(train_samples, batch_size=30)
validation_generator = generator(validation_samples, batch_size=30)

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

model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3, verbose = 1)

model.save('model06.h5')

from keras.models import load_model
new_model = load_model('model06.h5')
model.summary()
